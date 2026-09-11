"""스티커 이미지와 '얼굴 3D 좌표계의 어디에 붙일지'."""
import cv2
import numpy as np


class Sticker:
    """이미지 한 장과, 그것을 붙일 3D 사각형 하나."""

    def __init__(self, name, image, center, width_mm, local_yaw=0.0):
        self.name = name
        self.image = image                      # BGRA
        self.center = np.array(center, np.float64)   # 얼굴 모형 좌표(mm)
        self.width_mm = float(width_mm)
        self.local_yaw = float(local_yaw)       # 얼굴 옆면에 붙일 때 쓰는 국소 회전

    @property
    def aspect(self):
        h, w = self.image.shape[:2]
        return h / w

    def quad(self):
        """붙일 자리를 3D 사각형 네 점으로 만든다.

        순서는 이미지의 좌상·우상·우하·좌하와 같아야 한다. 이 순서가 어긋나면 스티커가
        뒤집히거나 대각선으로 꼬인 채 붙는다.
        """
        half_w = self.width_mm / 2
        half_h = self.width_mm * self.aspect / 2
        local = np.array([
            (-half_w, -half_h, 0.0),
            (half_w, -half_h, 0.0),
            (half_w, half_h, 0.0),
            (-half_w, half_h, 0.0),
        ], np.float64)

        # 볼처럼 얼굴 옆면에 붙는 스티커는 정면을 향한 평면에 두면 떠 보인다. y 축으로
        # 미리 돌려서 옆면에 눕힌다.
        if self.local_yaw:
            t = np.radians(self.local_yaw)
            rot = np.array([[np.cos(t), 0, np.sin(t)],
                            [0, 1, 0],
                            [-np.sin(t), 0, np.cos(t)]], np.float64)
            local = local @ rot.T

        return local + self.center


# 얼굴 모형 좌표 기준 배치표. 눈은 y=-32, 코끝은 (0,0,0), 입은 y=+28 이다.
# 2D 판본에서는 이 값들이 '눈 사이 거리의 몇 배'였는데, 여기서는 실제 mm 다. 얼굴이
# 멀어지거나 가까워지는 것은 투영이 알아서 처리하므로 배율을 곱할 필요가 없다.
LAYOUT = {
    'pirate_hat': dict(center=(0, -102, -30), width_mm=168),
    'eyepatch':   dict(center=(0, -32, -24), width_mm=118),
    'rednose':    dict(center=(0, 2, 8), width_mm=34),
    'mustache':   dict(center=(0, 15, -6), width_mm=76),
    # 볼은 랜드마크에 없다. 오른쪽 광대 위치를 모형 좌표로 직접 잡고, 얼굴 옆면을 향하도록
    # 눕힌다. 없는 부위를 아는 좌표계 안에서 만들어 쓰는 것이다.
    'scar':       dict(center=(-36, 4, -26), width_mm=34, local_yaw=-40),
}

ORDER = ['pirate_hat', 'eyepatch', 'rednose', 'mustache', 'scar']


def load(sticker_dir, names=None):
    """PNG 를 읽어 Sticker 목록을 만든다. 그리는 순서대로 돌려준다."""
    names = names or ORDER
    out = []
    for name in ORDER:
        if name not in names:
            continue
        path = f'{sticker_dir}/{name}.png'
        # imread 는 기본이 3채널이라 그냥 읽으면 알파가 조용히 사라지고, 합성했을 때
        # 스티커가 투명 없는 사각형 덩어리로 붙는다.
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f'스티커를 못 읽었다: {path}')
        if img.shape[2] != 4:
            raise SystemExit(f'알파 채널이 없다: {path} (4채널 PNG 여야 한다)')
        out.append(Sticker(name, img, **LAYOUT[name]))
    return out
