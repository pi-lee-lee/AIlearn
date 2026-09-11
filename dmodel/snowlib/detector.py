"""얼굴 검출과 랜드마크 5점 추출."""
import cv2
import numpy as np


class FaceDetector:
    """YuNet 래퍼. 한 줄 15개짜리 원시 출력을 쓰기 좋은 형태로 바꿔서 돌려준다."""

    def __init__(self, model_path, conf=0.5):
        # 기본 임계값 0.9 는 고개를 기울이거나 돌린 얼굴을 자주 놓친다. 자세를 다루는 것이
        # 목적이므로 낮춰 잡는다.
        self._det = cv2.FaceDetectorYN_create(model_path, '', (320, 320), conf)

    def detect(self, frame):
        h, w = frame.shape[:2]
        # 프레임 크기가 바뀔 수 있으니 매번 알려준다. 어긋나면 에러 없이 좌표만 틀어진다.
        self._det.setInputSize((w, h))
        _, raw = self._det.detect(frame)
        if raw is None:      # 얼굴이 없으면 빈 배열이 아니라 None 이 온다
            return []
        return [Face(row) for row in raw]


class Face:
    """한 줄 15개 = 상자 4 + 랜드마크 5쌍 10 + 점수 1."""

    # solvePnP 에 넘길 순서와 같아야 한다. headpose.MODEL_POINTS 와 짝이다.
    NAMES = ['r_eye', 'l_eye', 'nose', 'r_mouth', 'l_mouth']

    def __init__(self, row):
        self.box = row[:4].astype(int)          # x, y, w, h
        self.points = row[4:14].reshape(5, 2).astype(np.float64)
        self.score = float(row[14])

    def __getattr__(self, name):
        # face.nose 처럼 이름으로 꺼내 쓴다. row[8:10] 같은 숫자 슬라이스는 읽는 사람이
        # 매번 무슨 점인지 세어야 해서 틀리기 쉽다.
        if name in Face.NAMES:
            return self.points[Face.NAMES.index(name)]
        raise AttributeError(name)

    @property
    def eye_distance(self):
        return float(np.linalg.norm(self.l_eye - self.r_eye))
