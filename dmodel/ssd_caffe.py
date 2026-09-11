# 부트캠프 방식 — SSD 구조(res10) + Caffe 포맷을 cv2.dnn 으로 직접 조립한다.
#
# 이 폴더의 세 파일(ssd_caffe.py, yunet_onnx.py, yolo_ultra.py)은 아래 "카메라 루프"부터
# 파일 끝까지가 글자 하나까지 같다. 다른 것은 그 위의 detect() 뿐이다.
#
# 다만 코드가 같다고 셋이 같은 것을 찾지는 않는다. lena.jpg 로 재 보면 이 파일과 yunet 은
# 얼굴(화면의 11~12%)에 상자를 치고, yolo 는 사람 전신(63%)에 친다. 코드 구조가 같은 것과
# 찾는 대상이 같은 것은 별개의 축이다.
#
# 셋 중 이 방식만 전처리와 임계값을 손으로 쓴다. 포장을 안 뜯은 쪽이라 네 단계가 그대로
# 드러난다: 블롭 만들기 → 순전파 → 점수로 거르기 → 비율 좌표를 픽셀로 되돌리기.
import cv2

MODEL_NAME = 'res10-SSD/Caffe'
PROTOTXT = 'dmodel/models/deploy.prototxt'
WEIGHTS = 'dmodel/models/res10_ssd_fp16.caffemodel'
CONF = 0.5

# 설계도(prototxt)와 가중치(caffemodel)가 따로인 것이 Caffe 포맷의 특징이다.
# ONNX 는 둘을 한 파일에 담으므로 yunet 쪽은 인자가 하나다.
net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)


def detect(frame):
    h, w = frame.shape[:2]

    # 이 모델은 300x300 으로 줄인 그림만 본다. mean 을 빼는 것은 학습할 때 그렇게 정규화했기
    # 때문이고, 이 값이 틀리면 에러 없이 검출만 나빠진다.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    net.setInput(blob)
    det = net.forward()   # (1, 1, 200, 7) — 뒤의 7 은 [batch, class, conf, x1, y1, x2, y2]

    out = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < CONF:
            continue
        # 좌표가 0~1 비율로 나온다. 모델은 300x300 안에서만 보므로 원본 크기를 모르고,
        # 픽셀로 되돌리는 곱셈은 부르는 쪽 몫이다.
        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)
        out.append((x1, y1, x2, y2, conf, 'face'))
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
