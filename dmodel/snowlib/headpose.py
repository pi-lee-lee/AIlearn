"""랜드마크 5점에서 머리의 3D 자세를 푼다."""
import math

import cv2
import numpy as np

# 사람 얼굴의 대략적인 3D 좌표(mm). 코끝이 원점이고 축은 영상 좌표에 맞춘다.
#   +x 오른쪽, +y 아래, +z 카메라 쪽(코가 가장 튀어나옴)
# 내 얼굴의 실측이 아니라 일반 치수라, 각도의 방향과 대략의 크기는 맞지만 정밀값은 아니다.
# 순서는 detector.Face.NAMES 와 같아야 한다.
MODEL_POINTS = np.array([
    (-32.0, -32.0, -26.0),   # r_eye  (사람 기준 오른눈 = 이미지 왼쪽)
    (32.0, -32.0, -26.0),    # l_eye
    (0.0, 0.0, 0.0),         # nose
    (-28.0, 28.0, -22.0),    # r_mouth
    (28.0, 28.0, -22.0),     # l_mouth
], np.float64)


def camera_matrix(w, h):
    """카메라 내부 파라미터를 모를 때 쓰는 근사.

    초점거리를 이미지 폭으로 놓는다. 보정을 안 한 웹캠에서 흔히 쓰는 값이고, 이것 때문에
    각도 크기에 몇 도 오차가 생긴다. 방향과 상대 변화는 그대로 쓸 수 있다.
    """
    return np.array([[w, 0, w / 2],
                     [0, w, h / 2],
                     [0, 0, 1]], np.float64)


def solve(face, frame_shape):
    """(rvec, tvec, cam) 을 돌려준다. 실패하면 None."""
    h, w = frame_shape[:2]
    cam = camera_matrix(w, h)
    # SQPNP 를 쓰는 이유: 기본 ITERATIVE 는 점이 6개 이상이어야 한다. 우리는 5개뿐이다.
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, face.points, cam,
                                  np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    return rvec, tvec, cam


def euler_degrees(rvec):
    """사람이 읽을 수 있는 (pitch, yaw, roll) 도 단위. 화면 표시용이다.

    합성에는 쓰지 않는다 — 합성은 rvec 을 그대로 투영에 넘기는 쪽이 정확하고, 오일러각은
    분해 과정에서 부호나 축 순서를 헷갈리기 쉽다.
    """
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
    return pitch, yaw, roll


def face_scale(face):
    """모형 눈 사이 거리(64mm)에 대한 화면상 배율. 디버그 표시에만 쓴다."""
    return face.eye_distance / 64.0 if face.eye_distance else 0.0


def looks_valid(face, min_eye_px=8.0):
    """너무 작은 얼굴은 랜드마크가 몇 픽셀 안에서 흔들려 자세가 튄다."""
    return face.eye_distance >= min_eye_px and math.isfinite(face.eye_distance)
