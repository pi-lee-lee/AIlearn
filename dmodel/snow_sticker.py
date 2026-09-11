# 스노우 같은 스티커 효과. YuNet 랜드마크 5점 위에 해적 스티커를 얹는다.
#
# 같은 폴더의 세 파일과 달리 이 파일은 카메라부가 글자까지 같지는 않다. 그 셋은 상자를
# 그리는 것이 목적이라 draw 단계가 같았지만, 여기는 그리는 것이 아니라 합성이라 단계가
# 다르다. 카메라를 여닫고 통계를 내는 뼈대는 그대로 두었으니 나란히 놓고 보면 된다.
#
# 효과의 핵심은 랜드마크 좌표 자체가 아니라 거기서 뽑아내는 두 값이다.
#   크기 - 두 눈 사이 거리에 비례시킨다. 고정 크기로 붙이면 얼굴이 다가올 때 안 맞는다
#   각도 - 두 눈을 잇는 선의 기울기. 고개를 기울이면 스티커도 같이 기울어야 한다
# lena.jpg 기준으로 눈 사이 57.2px, 기울기 6.7도가 나온다. 모든 배치를 이 둘로 환산한다.
import math

import cv2
import numpy as np

MODEL_NAME = 'YuNet + 해적 스티커'
MODEL = 'opencv/yunet.onnx'
STICKER_DIR = 'dmodel/stickers'
CONF = 0.6

det_yn = cv2.FaceDetectorYN_create(MODEL, '', (320, 320), CONF)

# imread 는 기본값이 3채널이라 그냥 읽으면 알파가 조용히 사라지고, 합성했을 때 스티커가
# 사각형 덩어리로 붙는다. IMREAD_UNCHANGED 가 반드시 필요하다.
# 다섯 개를 한 얼굴에 다 붙이면 서로 겹쳐 복잡하다. PICK 으로 골라 쓴다.
#   PICK=eyepatch,mustache,scar .venv/bin/python yolo/dmodel/snow_sticker.py
import os

ALL = ['pirate_hat', 'eyepatch', 'rednose', 'mustache', 'scar']
PICK = [n for n in os.environ.get('PICK', ','.join(ALL)).split(',') if n in ALL]

STICKERS = {}
for _name in ALL:
    _img = cv2.imread(f'{STICKER_DIR}/{_name}.png', cv2.IMREAD_UNCHANGED)
    if _img is None:
        raise SystemExit(f'스티커를 못 읽었다: {STICKER_DIR}/{_name}.png')
    if _img.shape[2] != 4:
        raise SystemExit(f'알파 채널이 없다: {_name}.png (4채널 PNG 여야 한다)')
    STICKERS[_name] = _img


def overlay(bg, rgba, cx, cy, target_w, angle):
    """rgba 스티커를 bg 의 (cx, cy) 에 target_w 폭, angle 도로 얹는다."""
    h0, w0 = rgba.shape[:2]
    tw = max(int(target_w), 2)
    th = max(int(h0 * tw / w0), 2)
    # 줄일 때와 키울 때 좋은 보간이 다르다. 스티커를 원래 크게 만들어 두었으므로
    # 보통은 줄이는 쪽으로 간다.
    interp = cv2.INTER_AREA if tw < w0 else cv2.INTER_LINEAR
    small = cv2.resize(rgba, (tw, th), interpolation=interp)

    # 돌리면 네 귀퉁이가 캔버스 밖으로 나가 잘린다. 대각선 길이만큼 여백을 준 뒤 돌린다.
    diag = int(math.hypot(tw, th)) + 2
    canvas = np.zeros((diag, diag, 4), np.uint8)
    y0, x0 = (diag - th) // 2, (diag - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = small
    # 각도에 음수를 붙이는 이유: 영상 좌표는 y 가 아래로 증가해서 atan2 가 돌려주는 부호와
    # getRotationMatrix2D 가 도는 방향이 서로 반대다. 부호를 빼면 고개와 반대로 기울어진다.
    mat = cv2.getRotationMatrix2D((diag / 2, diag / 2), -angle, 1.0)
    canvas = cv2.warpAffine(canvas, mat, (diag, diag), flags=cv2.INTER_LINEAR,
                            borderValue=(0, 0, 0, 0))

    # 화면 밖으로 나가는 부분을 잘라낸다. 이 처리를 빼면 얼굴이 가장자리에 갔을 때
    # 배열 크기가 안 맞아 그대로 죽는다.
    x1, y1 = int(cx - diag / 2), int(cy - diag / 2)
    bx1, by1 = max(x1, 0), max(y1, 0)
    bx2, by2 = min(x1 + diag, bg.shape[1]), min(y1 + diag, bg.shape[0])
    if bx1 >= bx2 or by1 >= by2:
        return
    piece = canvas[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1]

    # 합성 자체는 이 두 줄이 전부다. 알파를 0~1 로 바꾼 뒤
    # 결과 = 스티커*a + 배경*(1-a) 를 픽셀마다 계산한다.
    alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
    roi = bg[by1:by2, bx1:bx2]
    bg[by1:by2, bx1:bx2] = (piece[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


def detect(frame):
    h, w = frame.shape[:2]
    det_yn.setInputSize((w, h))
    _, faces = det_yn.detect(frame)
    return [] if faces is None else list(faces)


def put_stickers(frame, face):
    """랜드마크 5점에서 붙일 자리·크기·각도를 만들어 스티커를 얹는다."""
    r_eye, l_eye = face[4:6], face[6:8]
    nose = face[8:10]
    r_mouth, l_mouth = face[10:12], face[12:14]

    eye_mid = (r_eye + l_eye) / 2
    mouth_mid = (r_mouth + l_mouth) / 2
    vec = l_eye - r_eye
    eye_dist = float(math.hypot(vec[0], vec[1]))
    if eye_dist < 8:            # 너무 작은 얼굴은 값이 튀어 스티커가 날아다닌다
        return
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))

    # 눈을 잇는 선과 직각인 '위쪽'. 고개가 기울면 이 방향도 같이 기울어서, 모자가 늘
    # 정수리를 향한다. 화면의 위쪽(0,-1)을 쓰면 고개를 기울일 때 모자가 어긋난다.
    up = np.array([vec[1], -vec[0]], np.float32) / eye_dist

    # 랜드마크에 볼은 없다. 눈과 입꼬리를 잇는 선의 중간에서 바깥으로 조금 밀면 볼이 된다.
    # 없는 부위는 있는 점들로 만들어 쓰는 것이고, 모자도 같은 방식이다.
    # 눈에서 입꼬리로 내려가는 선의 중간이 볼이다. 바깥으로 더 밀면 얼굴을 벗어나
    # 머리카락 위에 찍힌다(눈이 이미 얼굴 가장자리 쪽이라 여유가 없다).
    cheek = r_eye + (r_mouth - r_eye) * 0.55

    place = [
        ('pirate_hat', eye_mid + up * (eye_dist * 1.95), eye_dist * 3.2),
        ('eyepatch',   eye_mid,                          eye_dist * 2.7),
        ('rednose',    nose,                             eye_dist * 0.55),
        ('mustache',   nose + (mouth_mid - nose) * 0.52, eye_dist * 1.45),
        ('scar',       cheek,                            eye_dist * 0.72),
    ]
    for name, center, width in place:
        if name not in PICK:
            continue
        overlay(frame, STICKERS[name], float(center[0]), float(center[1]), width, angle)


# ===================== 카메라 루프 =====================
import time

SRC = os.environ.get('SRC', '0')
SHOW = os.environ.get('SHOW', '1') == '1'
SAVE = os.environ.get('SAVE', '')        # 마지막 합성 프레임을 파일로 남긴다(헤드리스 확인용)
MAX_SECONDS = float(os.environ.get('MAX_SECONDS', 30.0))

cap = cv2.VideoCapture(int(SRC) if SRC.isdigit() else SRC)
if not cap.isOpened():
    raise SystemExit(f'열 수 없다: {SRC}. 카메라라면 macOS 터미널에 카메라 권한이 필요하다.')

print('MODEL_NAME:', MODEL_NAME)
print('SRC:', SRC)
print('해상도:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frames = 0
infer_ms = []
face_frames = 0
start = time.time()
last = None

while time.time() - start < MAX_SECONDS:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frames += 1

    t0 = time.perf_counter()
    faces = detect(frame)
    infer_ms.append((time.perf_counter() - t0) * 1000)
    if faces:
        face_frames += 1
    for face in faces:
        put_stickers(frame, face)

    last = frame
    if SHOW:
        cv2.imshow(MODEL_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if SHOW:
    cv2.destroyAllWindows()
if SAVE and last is not None:
    # imwrite 는 경로가 없어도 예외를 던지지 않고 False 만 돌려준다. 확인하지 않으면
    # 저장이 안 된 줄 모르고 넘어간다.
    print('저장:', SAVE if cv2.imwrite(SAVE, last) else f'실패 - 경로를 확인하라: {SAVE}')

print('처리한 프레임:', frames)
if infer_ms:
    a = np.array(infer_ms[10:] if len(infer_ms) > 10 else infer_ms)
    print('평균 FPS:', round(frames / (time.time() - start), 1))
    print('검출 지연 평균(ms):', round(float(a.mean()), 1))
    print('얼굴이 잡힌 프레임:', f'{face_frames}/{frames}')
else:
    print('프레임을 하나도 읽지 못했다. SRC 를 확인하라.')
