"""3D 사각형을 화면에 투영하고, 원근 변환으로 스티커를 얹는다."""
import cv2
import numpy as np


def project(quad3d, rvec, tvec, cam):
    """얼굴 좌표계의 네 점을 화면 픽셀 좌표로 옮긴다."""
    pts, _ = cv2.projectPoints(quad3d, rvec, tvec, cam, np.zeros((4, 1)))
    return pts.reshape(-1, 2).astype(np.float32)


def facing_camera(dst):
    """투영된 사각형이 카메라를 향하고 있으면 True.

    고개를 많이 돌리면 스티커가 뒤통수 쪽으로 넘어가는데, 그때 네 점의 도는 방향이
    뒤집힌다. 부호로 걸러내지 않으면 뒤집힌 스티커가 얼굴에 겹쳐 그려진다.
    """
    x, y = dst[:, 0], dst[:, 1]
    # 신발끈 공식. 영상 좌표(y 아래)에서 좌상->우상->우하->좌하 순서는 양수가 된다.
    area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return area > 0


def paste(frame, rgba, dst):
    """rgba 를 dst 네 점이 만드는 사각형에 맞춰 원근 변환해 합성한다."""
    h, w = rgba.shape[:2]
    src = np.array([(0, 0), (w, 0), (w, h), (0, h)], np.float32)

    # 화면 전체 크기로 변환하면 매 프레임 수백만 픽셀을 헛돌린다. 스티커가 실제로 닿는
    # 사각형만 잘라서 그 안에서만 변환한다.
    x0, y0 = np.floor(dst.min(axis=0)).astype(int)
    x1, y1 = np.ceil(dst.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:      # 화면 밖으로 완전히 나갔다
        return

    mat = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], np.float32))
    warped = cv2.warpPerspective(rgba, mat, (x1 - x0, y1 - y0),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    # 합성은 이 두 줄이 전부다. 알파를 0~1 로 바꾸고 결과 = 스티커*a + 배경*(1-a).
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (warped[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


def render(frame, stickers, rvec, tvec, cam):
    """스티커 목록을 한 얼굴에 얹는다. 실제로 그린 개수를 돌려준다."""
    drawn = 0
    for sticker in stickers:
        dst = project(sticker.quad(), rvec, tvec, cam)
        if not np.isfinite(dst).all():
            continue
        if not facing_camera(dst):
            continue
        paste(frame, sticker.image, dst)
        drawn += 1
    return drawn


def draw_axes(frame, rvec, tvec, cam, length=60.0):
    """코끝에서 x(빨강)·y(초록)·z(파랑) 축을 그린다. 자세가 맞는지 눈으로 보는 용도."""
    axes = np.array([(0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length)], np.float64)
    pts, _ = cv2.projectPoints(axes, rvec, tvec, cam, np.zeros((4, 1)))
    p = pts.reshape(-1, 2).astype(int)
    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)], start=1):
        cv2.line(frame, tuple(p[0]), tuple(p[i]), color, 2)
