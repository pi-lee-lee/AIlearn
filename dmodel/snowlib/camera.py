"""카메라를 열고 프레임마다 처리 함수를 부르고 통계를 낸다."""
import os
import time

import cv2
import numpy as np


class Stats:
    def __init__(self, warmup=10):
        self.warmup = warmup
        self.frames = 0
        self.hit_frames = 0
        self.latency = []

    def add(self, ms, hit):
        self.frames += 1
        self.latency.append(ms)
        if hit:
            self.hit_frames += 1

    def report(self, elapsed):
        print('처리한 프레임:', self.frames)
        if not self.latency:
            print('프레임을 하나도 읽지 못했다. SRC 를 확인하라.')
            return
        # 앞 몇 프레임은 모델 워밍업이라 뺀다. 그보다 적으면 뺄 것이 없으므로 전부 쓴다.
        a = np.array(self.latency[self.warmup:] or self.latency)
        print('평균 FPS:', round(self.frames / elapsed, 1))
        print('처리 지연 평균(ms):', round(float(a.mean()), 1))
        print('처리 지연 p95(ms):', round(float(np.percentile(a, 95)), 1))
        print('얼굴이 잡힌 프레임:', f'{self.hit_frames}/{self.frames}')


def run(process, title):
    """process(frame) -> bool(얼굴을 잡았는가) 를 매 프레임 부른다.

    환경변수로 조종한다. 카메라는 권한과 사람 손이 필요해 자동으로 돌려볼 수 없으므로,
    파일이나 동영상을 대신 물릴 구멍(SRC)을 반드시 열어 둔다.
      SHOW=0 SRC=opencv/lena.jpg SAVE=out.png .venv/bin/python yolo/dmodel/snow3d.py
    """
    src = os.environ.get('SRC', '0')
    show = os.environ.get('SHOW', '1') == '1'
    save = os.environ.get('SAVE', '')
    max_seconds = float(os.environ.get('MAX_SECONDS', 30.0))

    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f'열 수 없다: {src}. 카메라라면 macOS 터미널에 카메라 권한이 필요하다.')

    print('제목:', title)
    print('SRC:', src)
    print('해상도:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    stats = Stats()
    start = time.time()
    last = None

    while time.time() - start < max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # 거울처럼 보이게 할 뿐 검출과는 무관하다

        t0 = time.perf_counter()
        hit = process(frame)
        stats.add((time.perf_counter() - t0) * 1000, hit)

        last = frame
        if show:
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if save and last is not None:
        # imwrite 는 경로가 없어도 예외 없이 False 만 돌려준다. 확인하지 않으면 저장이
        # 안 된 줄 모르고 넘어간다.
        print('저장:', save if cv2.imwrite(save, last) else f'실패 - 경로를 확인하라: {save}')

    stats.report(time.time() - start)
