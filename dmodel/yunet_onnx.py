# YuNet 구조 + ONNX 포맷을 FaceDetectorYN 으로 쓴다. 얼굴 전용 고수준 API 다.
#
# 이 폴더의 세 파일(ssd_caffe.py, yunet_onnx.py, yolo_ultra.py)은 아래 "카메라 루프"부터
# 파일 끝까지가 글자 하나까지 같다. 다른 것은 그 위의 detect() 뿐이다.
#
# 다만 코드가 같다고 셋이 같은 것을 찾지는 않는다. lena.jpg 로 재 보면 이 파일과 ssd 는
# 얼굴(화면의 11~12%)에 상자를 치고, yolo 는 사람 전신(63%)에 친다. 코드 구조가 같은 것과
# 찾는 대상이 같은 것은 별개의 축이다.
#
# ssd_caffe.py 가 손으로 쓰던 전처리·임계값·좌표 환산이 여기서는 전부 detect() 안쪽에 있다.
# 모델이 더 똑똑해져서가 아니라 같은 네 단계를 누가 대신 해주느냐가 다를 뿐이다.
# 5.0 이 추가한 신문물도 아니다 — 4.5.4(2021)부터 있었다.
import cv2

MODEL_NAME = 'YuNet/ONNX'
MODEL = 'opencv/yunet.onnx'
CONF = 0.6

# 셋째 인자가 입력 크기다. 아래에서 프레임마다 다시 넣으므로 여기 값은 자리를 채우는 용도다.
# 이 크기가 실제 프레임과 어긋나면 내부 환산이 틀어져, 에러 없이 상자만 엉뚱한 곳에 생긴다.
det_yn = cv2.FaceDetectorYN_create(MODEL, '', (320, 320), CONF)


def detect(frame):
    h, w = frame.shape[:2]

    # 카메라 해상도가 바뀔 수 있으니 매 프레임 알려준다. 미리 알려주기 때문에 아래 결과가
    # ssd 와 달리 비율이 아닌 픽셀 값으로 바로 나온다.
    det_yn.setInputSize((w, h))
    _, faces = det_yn.detect(frame)

    if faces is None:   # 얼굴이 없으면 빈 배열이 아니라 None 이 온다. 그냥 돌리면 TypeError 다.
        return []

    out = []
    for f in faces:
        # 한 줄이 15개다: 상자 4 + 눈·코·입 랜드마크 5쌍 10 + 점수 1.
        # 여기서는 상자만 쓰고 랜드마크는 버린다. ssd 와 출력을 맞추기 위해서다.
        x, y, bw, bh = f[:4].astype(int)
        out.append((x, y, x + bw, y + bh, float(f[-1]), 'face'))
    return out


# ===================== 카메라 루프 (세 파일 공통) =====================
# 여기부터 파일 끝까지는 세 파일이 글자까지 같다. 손으로 세 번 옮겨 적은 것이 아니라 한 번
# 써서 붙였고, 아래 명령으로 확인할 수 있다.
#
#   for f in yolo/dmodel/*.py; do awk '/^# =+ 카메라 루프/{p=1} p' "$f" | md5; done
#
# 해시 셋이 같으면 "모델이 달라도 영상 쪽 코드는 그대로"라는 말이 증명된 것이다.
import os
import time
import cv2

# 카메라는 권한과 사람 손이 필요해서 자동으로 돌려볼 수 없다. 그러면 세 파일이 정말 같은지
# 확인할 길이 없어지므로, 파일이나 동영상을 대신 물릴 구멍을 열어 둔다.
#   SHOW=0 SRC=opencv/lena.jpg .venv/bin/python yolo/dmodel/ssd_caffe.py
SRC = os.environ.get('SRC', '0')
SHOW = os.environ.get('SHOW', '1') == '1'
MAX_SECONDS = float(os.environ.get('MAX_SECONDS', 30.0))

cap = cv2.VideoCapture(int(SRC) if SRC.isdigit() else SRC)
if not cap.isOpened():
    raise SystemExit(f'열 수 없다: {SRC}. 카메라라면 macOS 터미널에 카메라 권한이 필요하다.')

print('MODEL_NAME:', MODEL_NAME)
print('SRC:', SRC)
print('해상도:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frames = 0
infer_ms = []
label_count = {}
start = time.time()

while time.time() - start < MAX_SECONDS:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)   # 거울처럼 보이게 하는 것뿐이고 검출 결과와는 무관하다
    frames += 1

    # 추론 시간만 따로 잰다. 읽기·그리기와 합쳐서 재면 모델을 바꾼 효과가 그 안에 묻힌다.
    t0 = time.perf_counter()
    boxes = detect(frame)
    infer_ms.append((time.perf_counter() - t0) * 1000)

    for x1, y1, x2, y2, conf, label in boxes:
        label_count[label] = label_count.get(label, 0) + 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f'{MODEL_NAME}  {infer_ms[-1]:.1f}ms  boxes={len(boxes)}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if SHOW:
        cv2.imshow(MODEL_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if SHOW:
    cv2.destroyAllWindows()

print('처리한 프레임:', frames)

if infer_ms:
    import numpy as np

    # 앞 열 프레임은 모델·카메라 워밍업이라 통계에서 뺀다. 프레임이 그보다 적으면 뺄 것이
    # 없으므로 전부 쓴다. 사진 한 장을 물렸을 때 여기서 빈 배열이 되는 것을 막는다.
    a = np.array(infer_ms[10:] if len(infer_ms) > 10 else infer_ms)
    print('평균 FPS:', round(frames / (time.time() - start), 1))
    print('추론 지연 평균(ms):', round(float(a.mean()), 1))
    print('추론 지연 p95(ms):', round(float(np.percentile(a, 95)), 1))
    print('추론 지연 최악(ms):', round(float(a.max()), 1))
    print('라벨별 상자 수:', label_count)
else:
    print('프레임을 하나도 읽지 못했다. SRC 를 확인하라.')
