# Face ID 식 얼굴 등록·판정 앱. 창 하나에 상태가 넷이다.
#   live    카메라 얼굴마다 등록된 이름 또는 unknown 을 붙인다. 하단에 [등록] [관리] 버튼.
#   name    등록할 이름을 키보드로 친다 (영문·숫자, Enter 확정, Esc 취소).
#   enroll  화면 가운데 원 안에서 고개를 돌리게 하고, 정면 + 8방향이 다 차면 저장한다.
#   manage  카메라를 끄고 등록된 사람 목록을 보여 준다. 줄을 골라 [삭제].
#
# 인식 파이프라인
#   YuNet(검출 + 눈·코·입 5점) → ArcFace R50(5점으로 정렬 후 512차원) → 등록된 중심과 코사인 비교
# 등록은 학습이 아니라 그 512개 숫자의 평균을 이름표와 저장하는 것이다. 자세한 사정은 embed() 참고.
#
# 인식망은 InsightFace buffalo_l 묶음 중 w600k_r50.onnx 한 파일만 쓴다(ResNet50, WebFace600K 학습).
# 처음엔 OpenCV 의 SFace(경량망, 128차원)였다. 같은 사진 18장(본인)과 lena(타인)로 재 보면
#   SFace    본인쌍 코사인 최소 0.510, lena-본인 코사인 최대 0.185
#   buffalo  본인쌍 코사인 최소 0.633, lena-본인 코사인 최대 0.044
# 본인끼리는 더 가깝고 타인과는 더 멀어 두 무리 사이 틈이 넓다. 틈이 넓을수록 문턱이 조금 어긋나도 판정이 안 뒤집힌다.
# 묶음 안의 SCRFD 검출기와 106점·68점 랜드마크 모델은 쓰지 않는다. 인식 정렬에는 5점이면 충분하다.
#
# 얼굴 각도는 3D 를 풀지 않고 2D 로 잰다. 코끝은 얼굴에서 가장 튀어나온 점이라, 고개를 돌리면 눈 둘·
# 입꼬리 둘이 만드는 사각형의 가운데에서 코끝이 돌린 쪽으로 빠져나온다. 그 오프셋을 눈 사이 거리로 나눈
# 값 d 가 각도 지표다. yolo/collect 프레임으로 재 보면 정면 0.00, 30도쯤 0.12, 60도쯤 0.5 다.
# snowlib.headpose 의 solvePnP 도 각도를 주지만 카메라 행렬을 짐작으로 넣고 일반 얼굴 치수를 쓰는 탓에
# 사람마다 몇 도씩 치우친다. 여기서는 정면을 그 사람의 0 으로 잡고(보정) 상대값만 쓰므로 그 문제가 없다.
#
# 진행률(%) 은 "얼마나 많은 방향을 보여 줬나" 다. 정면 1칸 + 8방향 8칸, 9칸이 차면 100%. 칸 하나는 그
# 방향에서 샘플 2장(0.15초 간격)이 모여야 찬다. 한 프레임 튄 것으로 칸이 차지 않게 하기 위해서다.
# 상하는 좌우보다 지표가 훨씬 덜 움직여서(PITCH_GAIN 주석 참고) 세로축에 이득을 곱한 타원 문턱을 쓴다.
# 샘플 수로 % 를 매기면 가만히 있어도 100% 가 되는데, 그렇게 등록한 사람은 고개를 조금만 돌려도
# unknown 이 된다. 등록 화면이 굳이 고개를 돌리라고 시키는 이유가 그것이다.
#
#   .venv/bin/python faceid/faceID.py                 # 카메라 0
#   SRC=clip.mp4 .venv/bin/python faceid/faceID.py    # 동영상으로 대신
# 키: r 등록, m 관리, q 종료.
import os
import glob
import time
import cv2
import numpy as np

MODEL_NAME = 'YuNet/ONNX + ArcFace-R50(buffalo_l)/ONNX'
DET_MODEL = 'faceid//models/yunet.onnx'
REC_MODEL = 'faceid/models/w600k_r50.onnx'
REC_DIM = 512
FACES_DIR = 'faceid/faces'
WIN = 'faceID'
MAX_W = 1280      # 이보다 넓은 프레임은 줄인다. 1080p 카메라를 그대로 띄우면 노트북 화면을 넘고 검출도 느리다
CONF = 0.6        # 검출 문턱. yunet_onnx.py 와 같다
REG_CONF = 0.8    # 등록에 쓸 얼굴은 더 확실한 것만. 흐릿한 샘플이 섞이면 중심이 흔들린다
# 같은 사람으로 볼 코사인 문턱. SFace 때 쓰던 0.363 은 SFace 전용 값이라 옮겨 오지 않았다.
# 위 측정에서 본인쌍 최소 0.633 과 lena 최대 0.044 의 한가운데쯤을 잡았다. 타인이 lena 한 명뿐인
# 측정이라 검증된 값은 아니다. 등록 때 찍히는 "샘플-중심 코사인 최소" 가 이보다 넉넉히 위인지 보라.
THRESH = float(os.environ.get('THRESH', 0.35))

if not os.path.exists(REC_MODEL):
    raise SystemExit(f'{REC_MODEL} 이 없다. https://github.com/deepinsight/insightface/releases/download/'
                     'v0.7/buffalo_l.zip 을 받아 그 안의 w600k_r50.onnx 만 faceid/models/ 에 두라.')

det_yn = cv2.FaceDetectorYN_create(DET_MODEL, '', (320, 320), CONF)
rec_net = cv2.dnn.readNetFromONNX(REC_MODEL)

# ArcFace 가 학습할 때 쓴 112x112 정면 틀. 순서대로 눈·눈·코·입꼬리·입꼬리가 이 자리에 오도록 사진을
# 돌리고 줄인다. 학습 때와 다른 틀에 맞추면 망은 처음 보는 배치의 얼굴을 받게 되어 코사인이 조용히 떨어진다.
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


# ===================== 인식 =====================

def detect(frame):
    """YuNet 원시 출력을 그대로 돌려준다. 한 줄 15개 = 상자 4 + 랜드마크 5쌍 10 + 점수 1."""
    h, w = frame.shape[:2]
    det_yn.setInputSize((w, h))
    _, faces = det_yn.detect(frame)
    return [] if faces is None else list(faces)


def embed(frame, face):
    """얼굴 한 줄(15개) → 길이 1 로 맞춘 512차원 벡터."""
    # 인식망은 수십만 명의 얼굴로 "같은 사람이면 가깝게, 다른 사람이면 멀게" 이미 학습돼 있다.
    # 여기서 가중치가 바뀌는 일은 없고, 등록은 이 벡터들의 평균을 저장하는 것이 전부다.
    # 랜드마크 5점을 ARCFACE_DST 에 맞추는 회전·크기·이동(값 4개)을 구해 112x112 로 옮긴다. 값 4개에
    # 점 5개(식 10개)라 남는 식으로 튀는 점 하나를 걸러낼 수 있어서 LMEDS 를 쓴다. 정렬 없이 상자만
    # 잘라 넣으면 같은 사람도 각도에 따라 코사인이 크게 떨어진다.
    M, _ = cv2.estimateAffinePartial2D(face[4:14].reshape(5, 2), ARCFACE_DST, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (112, 112))
    # 학습 때 입력이 RGB, 화소값 (x - 127.5) / 127.5 였다. OpenCV 사진은 BGR 이라 swapRB 로 뒤집는다.
    rec_net.setInput(cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True))
    feat = rec_net.forward().ravel()
    return feat / np.linalg.norm(feat)   # 길이 1 이면 코사인 = 내적


def load_registry():
    """faceid/faces/*.npy → {이름: {'center': 길이 1 중심, 'n': 샘플 수, 'mtime': 등록 시각}}."""
    reg = {}
    for p in sorted(glob.glob(os.path.join(FACES_DIR, '*.npy'))):
        e = np.load(p)
        # SFace 시절 등록 파일은 128차원이라 512차원과 내적이 안 된다. 사진이 아니라 벡터만 저장했으므로
        # 변환할 방법이 없고 다시 등록해야 한다.
        if e.shape[1] != REC_DIM:
            raise SystemExit(f'{p} 는 {e.shape[1]}차원이다(지금 인식망은 {REC_DIM}). 파일을 치우고 다시 등록하라.')
        c = e.mean(axis=0)
        reg[os.path.splitext(os.path.basename(p))[0]] = {
            'center': c / np.linalg.norm(c), 'n': len(e), 'mtime': os.path.getmtime(p)}
    return reg


def identify(e, reg):
    """가장 가까운 중심의 이름과 그 코사인. 코사인이 문턱 미만이면 unknown."""
    names = list(reg)
    sims = np.array([reg[n]['center'] @ e for n in names])
    k = int(sims.argmax())
    return (names[k] if sims[k] >= THRESH else 'unknown'), float(sims[k])


def head_dir(face):
    """코끝이 눈·입 네 점의 가운데에서 빠져나간 방향·크기 d. 눈 사이 거리로 나눠 얼굴 크기와 무관하다."""
    r_eye, l_eye, nose, r_mouth, l_mouth = face[4:14].reshape(5, 2)
    outer = (r_eye + l_eye + r_mouth + l_mouth) / 4
    eye = l_eye - r_eye
    d = (nose - outer) / np.linalg.norm(eye)
    # 고개를 옆으로 기울이면(roll) 눈선이 기울고 d 도 같이 돈다. 눈선을 수평으로 되돌린 좌표계에서 잰다.
    roll = np.arctan2(eye[1], eye[0])
    c, s = np.cos(-roll), np.sin(-roll)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


# ===================== 등록 =====================

N_SECTORS = 8          # 원 둘레를 45도씩 여덟 칸
CENTER_N = 5           # 정면 샘플 수. 이 샘플의 d 평균이 그 사람의 0점이다
SECTOR_N = 2           # 칸 하나를 채우는 샘플 수
SAMPLE_GAP = 0.15      # 샘플 사이 최소 간격(초). 이웃 프레임은 거의 같은 얼굴이라 붙여 찍어도 정보가 없다
CENTER_MAX = 0.07      # |d| 가 이 안이면 정면
TILT_MIN = 0.12        # 칸을 채우려면 이만큼은 돌려야 한다 (30도 안팎)
TILT_MAX = 0.45        # 이보다 돌리면 랜드마크가 흔들려 정렬이 틀어진다
# 세로 이득. 이론상 코끝 오프셋은 좌우·상하에 같은 크기로 반응해야 하지만 실제로는 상하가 훨씬 약하다.
# 노트북 카메라 앞에서 고개를 30도 숙이면 눈이 가려져 검출이 흔들리고, YuNet 의 코 랜드마크도 세로로는
# 덜 움직인다. 그래서 dy 에 이득을 곱해 문턱을 타원으로 만든다 — 위아래는 좌우의 1/2.5 만 움직여도
# 칸이 찬다. 끄덕여도 흰 점이 안 올라오면 PITCH_GAIN=3.5 처럼 키우고, 튀면 줄인다.
PITCH_GAIN = float(os.environ.get('PITCH_GAIN', 2.5))
# 화면 방향 이름. sector_of() 와 같은 순서(0=오른쪽부터 시계 방향). 거울 화면이라 사용자 기준과 같다.
SECTOR_NAMES = [('오른쪽', 'right'), ('오른쪽 아래', 'down-right'), ('아래', 'down'), ('왼쪽 아래', 'down-left'),
                ('왼쪽', 'left'), ('왼쪽 위', 'up-left'), ('위', 'up'), ('오른쪽 위', 'up-right')]


def sector_of(rel):
    """화면 좌표(x 오른쪽, y 아래)에서 0=오른쪽부터 시계 방향 45도씩."""
    ang = np.degrees(np.arctan2(rel[1], rel[0]))
    return int(np.round(ang / 45.0)) % N_SECTORS


class Enroller:
    def __init__(self, name):
        self.name = name
        self.center = []                          # 정면 임베딩
        self.center_d = []                        # 정면일 때의 d
        self.sectors = [[] for _ in range(N_SECTORS)]
        self.last = 0.0
        self.msg = 'in_circle'
        self.raw = None                           # 이번 프레임의 d (표시용)
        self.rel = None                           # (d - 0점) 에 세로 이득을 곱한 것. 칸 판정과 흰 점은 이것을 쓴다

    @property
    def zero(self):
        return np.mean(self.center_d, axis=0) if self.center_d else np.zeros(2)

    @property
    def progress(self):
        done = min(len(self.center), CENTER_N) / CENTER_N
        done += sum(min(len(s), SECTOR_N) / SECTOR_N for s in self.sectors)
        return done / (1 + N_SECTORS)

    def samples(self):
        return np.stack(self.center + [e for s in self.sectors for e in s])

    def missing(self):
        return [k for k, s in enumerate(self.sectors) if len(s) < SECTOR_N]

    def update(self, frame, faces, circle):
        cx, cy, R = circle
        self.raw = self.rel = None
        if not faces:
            self.msg = 'in_circle'
            return
        f = max(faces, key=lambda r: r[2] * r[3])   # 여럿이면 가장 큰(=가까운) 얼굴이 등록 대상
        x, y, w, h = f[:4]
        off_center = np.hypot(x + w / 2 - cx, y + h / 2 - cy)
        if off_center > 0.45 * R or w < 0.6 * R or w > 2.4 * R or f[14] < REG_CONF:
            self.msg = 'in_circle'
            return

        d = head_dir(f)
        self.raw = d
        now = time.time()
        if len(self.center) < CENTER_N:
            # 정면 단계. 0점을 잡는 중이라 코가 가운데 가까이 있을 때만 받는다.
            self.msg = 'look_straight'
            self.rel = d
            if np.linalg.norm(d) < CENTER_MAX and now - self.last >= SAMPLE_GAP:
                self.center.append(embed(frame, f))
                self.center_d.append(d)
                self.last = now
            return

        self.rel = (d - self.zero) * np.array([1.0, PITCH_GAIN])
        mag = np.linalg.norm(self.rel)
        if mag > TILT_MAX:
            self.msg = 'too_far'
            return
        self.msg = 'turn'
        if mag >= TILT_MIN:
            k = sector_of(self.rel)
            if len(self.sectors[k]) < SECTOR_N and now - self.last >= SAMPLE_GAP:
                self.sectors[k].append(embed(frame, f))
                self.last = now

    def save(self, registry):
        e = self.samples()
        c = e.mean(axis=0)
        c /= np.linalg.norm(c)
        self_sim = e @ c
        print('등록 완료:', self.name)
        print('등록 샘플 수:', len(e))
        print('샘플-중심 코사인 최소:', round(float(self_sim.min()), 3))
        print('샘플-중심 코사인 평균:', round(float(self_sim.mean()), 3))
        others = {n: round(float(registry[n]['center'] @ c), 3) for n in registry if n != self.name}
        if others:
            # 문턱보다 낮아야 두 사람이 갈린다. 위면 verify 가 둘을 헷갈린다.
            print('다른 등록자 중심과의 코사인:', others)
            for n, s in others.items():
                if s >= THRESH:
                    print(f'주의: {n} 와 코사인 {s} 로 문턱 {THRESH} 이상이다. 둘을 구분하지 못한다.')
        os.makedirs(FACES_DIR, exist_ok=True)
        path = os.path.join(FACES_DIR, f'{self.name}.npy')
        np.save(path, e)   # 중심이 아니라 샘플을 저장한다. 문턱이나 규칙을 바꿔도 다시 찍을 필요가 없다
        print('저장:', path, e.shape)


# ===================== 화면 =====================

# cv2.putText 의 Hershey 글꼴은 ASCII 뿐이라 한글은 PIL 로 한 번 그려 캐시한 뒤 얹는다.
# 한글 글꼴이 없는 환경이면 영문 문구로 내려간다.
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
              '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
              '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
FONT = next((p for p in FONT_PATHS if os.path.exists(p)), None)
STR = {   # (한글, 영문 대체)
    'register': ('등록', 'REGISTER'), 'manage': ('관리', 'MANAGE'), 'cancel': ('취소', 'CANCEL'),
    'delete': ('삭제', 'DELETE'), 'back': ('뒤로', 'BACK'), 'ok': ('확인', 'OK'),
    'name_prompt': ('등록할 이름 (영문·숫자) 를 치고 Enter', 'Type a name (a-z, 0-9) and press Enter'),
    'in_circle': ('얼굴을 원 안에 맞추세요', 'Fit your face in the circle'),
    'look_straight': ('카메라를 정면으로 보세요', 'Look straight at the camera'),
    'turn': ('고개를 천천히 돌려 둘레를 채우세요', 'Slowly turn your head to fill the ring'),
    'too_far': ('너무 많이 돌렸습니다. 조금만 돌아오세요', 'Too far. Come back a little'),
    'done': ('등록 완료', 'Registered'), 'remaining': ('남은 방향', 'Remaining'),
    'users': ('등록된 사용자', 'Registered users'), 'none': ('등록된 사용자가 없습니다', 'No users yet'),
    'samples': ('샘플', 'samples'),
}
_fonts, _sprites = {}, {}


def S(key):
    return STR[key][0 if FONT else 1]


def text(img, s, x, y, size=22, color=(255, 255, 255), anchor='left'):
    """img 에 글자를 얹고 (폭, 높이) 를 돌려준다. anchor 는 x 기준 left/center/right."""
    if FONT is None:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        x -= {'left': 0, 'center': tw // 2, 'right': tw}[anchor]
        cv2.putText(img, s, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 2)
        return tw, th
    key = (s, size, color)
    if key not in _sprites:
        if size not in _fonts:
            _fonts[size] = ImageFont.truetype(FONT, size)
        l, t, r, b = _fonts[size].getbbox(s)
        im = Image.new('RGBA', (r + 2, b + 2), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), s, font=_fonts[size], fill=(color[2], color[1], color[0], 255))
        _sprites[key] = np.array(im)
    sp = _sprites[key]
    sh, sw = sp.shape[:2]
    x -= {'left': 0, 'center': sw // 2, 'right': sw}[anchor]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + sw, W), min(y + sh, H)
    if x1 <= x0 or y1 <= y0:
        return sw, sh
    crop = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    a = crop[..., 3:4] / 255.0
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + crop[..., 2::-1] * a).astype(np.uint8)   # PIL 은 RGB, cv2 는 BGR
    return sw, sh


BAR_H = 64
BTN_W, BTN_H = 150, 44


class Button:
    def __init__(self, key, x, y, w=BTN_W, h=BTN_H, color=(70, 70, 70)):
        self.key, self.x, self.y, self.w, self.h, self.color = key, x, y, w, h, color

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 1)
        text(img, S(self.key), self.x + self.w // 2, self.y + 9, 22, anchor='center')

    def hit(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def bottom_bar(img, keys):
    """하단 반투명 바에 버튼을 가운데 정렬로 늘어놓고 돌려준다."""
    H, W = img.shape[:2]
    bar = img[H - BAR_H:H]
    bar[:] = (bar * 0.35).astype(np.uint8)
    gap = 24
    total = len(keys) * BTN_W + (len(keys) - 1) * gap
    x = (W - total) // 2
    btns = []
    for k in keys:
        color = (40, 40, 160) if k == 'delete' else (70, 70, 70)
        btns.append(Button(k, x, H - BAR_H + (BAR_H - BTN_H) // 2, color=color))
        x += BTN_W + gap
    for b in btns:
        b.draw(img)
    return btns


def circle_of(shape):
    """등록용 원. 하단 바를 뺀 영역의 가운데."""
    H, W = shape[:2]
    R = int(min(W, H - BAR_H) * 0.30)
    return W // 2, (H - BAR_H) // 2, R


def draw_enroll(img, enr):
    cx, cy, R = circle_of(img.shape)
    # Face ID 처럼 원 바깥을 어둡게 해서 시선을 원 안으로 모은다
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (cx, cy), R, 255, -1)
    dark = (img * 0.3).astype(np.uint8)
    img[:] = cv2.bitwise_and(img, img, mask=mask) + cv2.bitwise_and(dark, dark, mask=cv2.bitwise_not(mask))
    cv2.circle(img, (cx, cy), R, (255, 255, 255), 1)

    # 둘레의 8칸. 찬 칸은 초록, 반쯤 찬 칸은 노랑, 빈 칸은 회색. cv2.ellipse 의 각도는 x축에서 시계 방향이라
    # sector_of() 와 같은 규약이다.
    for k, s in enumerate(enr.sectors):
        color = (0, 200, 0) if len(s) >= SECTOR_N else (0, 200, 255) if s else (90, 90, 90)
        a0 = 45 * k - 22.5 + 3
        cv2.ellipse(img, (cx, cy), (R + 16, R + 16), 0, a0, a0 + 45 - 6, color, 8)
    # 정면 칸은 원 중앙의 점
    n = len(enr.center)
    cv2.circle(img, (cx, cy), 6, (0, 200, 0) if n >= CENTER_N else (0, 200, 255) if n else (90, 90, 90), -1)
    # 지금 코가 향한 방향. 사용자가 자기 움직임과 칸의 대응을 바로 보게 한다.
    if enr.rel is not None:
        mag = min(np.linalg.norm(enr.rel) / TILT_MAX, 1.0)
        u = enr.rel / (np.linalg.norm(enr.rel) + 1e-9)
        cv2.circle(img, (int(cx + u[0] * mag * R), int(cy + u[1] * mag * R)), 7, (255, 255, 255), -1)

    text(img, S(enr.msg), cx, 24, 26, anchor='center')
    if enr.msg == 'turn' and enr.missing():
        # 남은 방향을 말로도 알려 준다. 위·아래는 흰 점만 보고는 얼마나 더 움직여야 할지 감이 안 온다.
        names = ', '.join(SECTOR_NAMES[k][0 if FONT else 1] for k in enr.missing())
        text(img, f'{S("remaining")}: {names}', cx, 58, 20, (0, 200, 255), anchor='center')
    text(img, f'{int(enr.progress * 100)}%', cx, cy + R + 34, 34, anchor='center')
    text(img, enr.name, cx, cy + R + 76, 22, (200, 200, 200), anchor='center')
    if enr.raw is not None:
        # 원시 지표. PITCH_GAIN 을 맞출 때 이 숫자를 본다. 정면 0.00, 좌우 30도쯤 dx 0.12.
        text(img, f'dx {enr.raw[0]:+.2f}  dy {enr.raw[1]:+.2f}  gain x{PITCH_GAIN:g}',
             12, img.shape[0] - BAR_H - 30, 18, (160, 160, 160))


# ===================== 앱 =====================

class App:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.registry = load_registry()
        self.state = 'live'
        self.enr = None
        self.typed = ''
        self.selected = None
        self.clicks = []
        self.buttons = []
        self.rows = []
        self.done_until = 0.0
        self.size = (720, 1280)
        self.frames = 0
        self.infer_ms = []
        self.label_count = {}

    # ----- 카메라 -----
    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(int(self.src) if self.src.isdigit() else self.src)
            if not self.cap.isOpened():
                raise SystemExit(f'열 수 없다: {self.src}. 카메라라면 macOS 터미널에 카메라 권한이 필요하다.')
            print('해상도:', int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x',
                  int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close_camera(self):
        # 관리 화면에서는 카메라를 끈다. 켜 둘 이유가 없고, 꺼졌다는 것이 표시등으로도 보인다.
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ----- 상태 전이 -----
    def go(self, state):
        if state == 'name':
            self.typed = 'owner' if not self.registry else f'user{len(self.registry) + 1}'
        elif state == 'enroll':
            self.enr = Enroller(self.typed)
            print('등록 시작:', self.typed)
        elif state == 'manage':
            self.close_camera()
            self.selected = None
        elif state == 'live':
            self.open_camera()
        self.state = state

    def on_click(self, px, py):
        for b in self.buttons:
            if b.hit(px, py):
                self.on_button(b.key)
                return
        if self.state == 'manage':
            for name, (y0, y1) in self.rows:
                if y0 <= py <= y1:
                    self.selected = name

    def on_button(self, key):
        if key == 'register':
            self.go('name')
        elif key == 'manage':
            self.go('manage')
        elif key == 'ok' and self.typed:
            self.go('enroll')
        elif key in ('cancel', 'back'):
            if self.state == 'enroll':
                print('등록 취소:', self.enr.name)
            self.go('live')
        elif key == 'delete' and self.selected:
            os.remove(os.path.join(FACES_DIR, f'{self.selected}.npy'))
            print('삭제:', self.selected)
            self.registry = load_registry()
            self.selected = None

    def on_key(self, k):
        if k in (-1, 255):   # 눌린 키가 없으면 -1 이고, & 0xFF 를 거치면 255 다
            return True
        if self.state == 'name':
            if k == 13:
                self.on_button('ok')
            elif k == 27:
                self.go('live')
            elif k in (8, 127):
                self.typed = self.typed[:-1]
            elif k < 128 and len(self.typed) < 16 and (chr(k).isalnum() or chr(k) in '_-'):
                self.typed += chr(k)   # 파일 이름이 되므로 영문·숫자·_- 만 받는다
            return True
        if k == ord('q'):
            return False
        if self.state == 'live' and k == ord('r'):
            self.go('name')
        elif self.state == 'live' and k == ord('m'):
            self.go('manage')
        elif k == 27 and self.state in ('enroll', 'manage'):
            self.on_button('cancel')
        return True

    # ----- 프레임 하나 -----
    def step(self):
        """프레임 하나를 읽고 그려서 돌려준다. 영상이 끝나면 None."""
        while self.clicks:
            self.on_click(*self.clicks.pop(0))

        if self.state == 'manage':
            img = np.full((*self.size, 3), 30, np.uint8)
            self.render_manage(img)
            return img

        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)   # 거울처럼 보이게 할 뿐. 등록과 판정이 같이 뒤집히므로 결과에는 영향 없다
        if frame.shape[1] > MAX_W:
            frame = cv2.resize(frame, (MAX_W, frame.shape[0] * MAX_W // frame.shape[1]))
        self.size = frame.shape[:2]
        self.frames += 1

        if self.state == 'live':
            self.render_live(frame)
        elif self.state == 'name':
            self.render_name(frame)
        elif self.state == 'enroll':
            self.render_enroll(frame)
        return frame

    def render_live(self, img):
        t0 = time.perf_counter()
        results = [(f,) + identify(embed(img, f), self.registry) for f in detect(img)] if self.registry \
            else [(f, 'unknown', 0.0) for f in detect(img)]
        self.infer_ms.append((time.perf_counter() - t0) * 1000)
        for f, name, sim in results:
            self.label_count[name] = self.label_count.get(name, 0) + 1
            x, y, w, h = f[:4].astype(int)
            color = (0, 0, 255) if name == 'unknown' else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{name} {sim:.2f}', (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for i in range(5):
                cv2.circle(img, (int(f[4 + 2 * i]), int(f[5 + 2 * i])), 2, (0, 255, 255), -1)
        cv2.putText(img, f'{MODEL_NAME}  {self.infer_ms[-1]:.1f}ms  faces={len(results)}  users={len(self.registry)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if time.time() < self.done_until:
            text(img, f'{S("done")}: {self.enr.name}', img.shape[1] // 2, 40, 30, (0, 255, 0), anchor='center')
        self.buttons = bottom_bar(img, ['register', 'manage'])

    def render_name(self, img):
        img[:] = (img * 0.4).astype(np.uint8)
        H, W = img.shape[:2]
        text(img, S('name_prompt'), W // 2, H // 2 - 70, 24, anchor='center')
        cv2.rectangle(img, (W // 2 - 180, H // 2 - 24), (W // 2 + 180, H // 2 + 24), (255, 255, 255), 1)
        caret = '|' if int(time.time() * 2) % 2 == 0 else ' '
        text(img, self.typed + caret, W // 2, H // 2 - 14, 28, anchor='center')
        self.buttons = bottom_bar(img, ['ok', 'cancel'])

    def render_enroll(self, img):
        self.enr.update(img, detect(img), circle_of(img.shape))
        draw_enroll(img, self.enr)
        self.buttons = bottom_bar(img, ['cancel'])
        if self.enr.progress >= 1.0:
            self.enr.save(self.registry)
            self.registry = load_registry()
            self.done_until = time.time() + 2.0
            self.go('live')

    def render_manage(self, img):
        H, W = img.shape[:2]
        text(img, S('users'), W // 2, 30, 30, anchor='center')
        self.rows = []
        y = 100
        if not self.registry:
            text(img, S('none'), W // 2, y, 24, (150, 150, 150), anchor='center')
        for name, info in self.registry.items():
            y0, y1 = y, y + 44
            if name == self.selected:
                cv2.rectangle(img, (W // 2 - 300, y0), (W // 2 + 300, y1), (40, 40, 160), -1)
            text(img, name, W // 2 - 280, y0 + 8, 26)
            when = time.strftime('%Y-%m-%d %H:%M', time.localtime(info['mtime']))
            text(img, f'{info["n"]} {S("samples")}   {when}', W // 2 + 280, y0 + 12, 20, (180, 180, 180), anchor='right')
            self.rows.append((name, (y0, y1)))
            y += 52
        self.buttons = bottom_bar(img, ['delete', 'back'])

    # ----- 루프 -----
    def run(self):
        self.open_camera()
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, lambda ev, x, y, flags, _: self.clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
        start = time.time()
        while True:
            img = self.step()
            if img is None:
                break
            cv2.imshow(WIN, img)
            if not self.on_key(cv2.waitKey(1) & 0xFF):
                break
        self.close_camera()
        cv2.destroyAllWindows()
        print('처리한 프레임:', self.frames)
        if self.infer_ms:
            a = np.array(self.infer_ms[10:] if len(self.infer_ms) > 10 else self.infer_ms)
            print('평균 FPS:', round(self.frames / (time.time() - start), 1))
            print('검출+인식 지연 평균(ms):', round(float(a.mean()), 1))
            print('판정별 얼굴 수:', self.label_count)


if __name__ == '__main__':
    print('MODEL_NAME:', MODEL_NAME)
    print('문턱(THRESH):', THRESH)
    print('한글 글꼴:', FONT or '없음 (영문으로 표시)')
    app = App(os.environ.get('SRC', '0'))
    print('등록된 사람:', list(app.registry))
    app.run()
