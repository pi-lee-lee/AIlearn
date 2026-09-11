# YOLO11n 구조 + PyTorch(.pt) 포맷을 ultralytics 로 쓴다.
#
# 이 폴더의 세 파일(ssd_caffe.py, yunet_onnx.py, yolo_ultra.py)은 아래 "카메라 루프"부터
# 파일 끝까지가 글자 하나까지 같다. 다른 것은 그 위의 detect() 뿐이다.
#
# 다만 코드가 같다고 셋이 같은 것을 찾지는 않는다. lena.jpg 로 재 보면 ssd 와 yunet 은
# 얼굴(화면의 11~12%)에 상자를 치는데 이 파일은 사람 전신(63%)에 친다. 코드 구조가 같은
# 것과 찾는 대상이 같은 것은 별개의 축이고, 후자를 정하는 것은 가중치다.
#
# 포맷도 셋 중 이것만 OpenCV 가 읽지 못한다. .pt 는 PyTorch 체크포인트라 cv2.dnn 으로는
# 못 열고, 열려면 ONNX 로 내보내야 한다. 그래서 여기만 cv2 대신 ultralytics 를 쓴다.
from ultralytics import YOLO

MODEL_NAME = 'YOLO11n/Ultralytics'
MODEL = 'yolo11n.pt'
CONF = 0.4

model = YOLO(MODEL)

# 이 가중치가 아는 80종(COCO)에 'face' 는 없고 'person' 만 있다. 그래서 얼굴이 아니라
# 사람 전신에 상자가 쳐진다. 나머지 79종까지 그리면 비교가 흐려지므로 사람만 남긴다.
PERSON = [i for i, n in model.names.items() if n == 'person']


def detect(frame):
    # 전처리·임계값·NMS 가 전부 predict() 안에 있다. FaceDetectorYN 과 같은 높이의 API 라
    # 코드 모양이 ssd 쪽보다 yunet 쪽을 닮은 것이고, 모델 계열과는 상관이 없다.
    r = model.predict(frame, conf=CONF, classes=PERSON, verbose=False)[0]

    out = []
    for b in r.boxes:
        # xyxy 는 좌상단·우하단 픽셀이다. 학습용 YOLO 라벨 포맷(중심점 + 0~1 정규화)과는
        # 표현이 다르므로 라벨 파일을 만들 때는 따로 변환해야 한다.
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        out.append((x1, y1, x2, y2, float(b.conf), model.names[int(b.cls)]))
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
