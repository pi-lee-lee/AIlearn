# 스노우식 해적 스티커 — 3D 자세 판본.
#
# snow_sticker.py 와 하는 일은 같지만 붙이는 방식이 다르다.
#   2D 판본  스티커를 화면 좌표에 놓고 눈선 기울기(roll)만큼 돌린다.
#            고개를 옆으로 돌려도 스티커는 정면 모양 그대로라 평면으로 떠 보인다.
#   3D 판본  스티커를 얼굴 3D 좌표계에 놓고 화면으로 투영한다.
#            yaw·pitch·원근이 투영 계산에서 저절로 따라온다.
#
# lena.jpg 는 어깨 너머로 돌아보는 사진이라 yaw 가 50도쯤 된다. 두 판본을 같은 사진으로
# 돌려 비교하면 차이가 바로 보인다.
#
# 코드는 기능별로 snowlib/ 에 나눠 두었다. 이 파일은 그것들을 엮는 일만 한다.
import os
import sys

# 프로젝트 루트에서 실행하므로 snowlib 가 import 경로에 없다. 이 파일이 있는 폴더를
# 넣어 준다. 루트에서 돌리는 저장소 관례를 깨지 않으면서 묶음을 나누기 위한 처리다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snowlib import compose, detector, headpose, stickers
from snowlib import camera

MODEL = 'opencv/yunet.onnx'
STICKER_DIR = 'dmodel/stickers'
TITLE = 'YuNet 3D + 해적 스티커'

# 다섯 개를 다 붙이면 서로 겹쳐 복잡하다. PICK 으로 골라 쓴다.
#   PICK=eyepatch,mustache,scar .venv/bin/python yolo/dmodel/snow3d.py
PICK = [n for n in os.environ.get('PICK', ','.join(stickers.ORDER)).split(',')
        if n in stickers.ORDER]
# AXES=1 이면 코끝에 좌표축을 그린다. 자세가 제대로 풀렸는지 눈으로 확인할 때 쓴다.
SHOW_AXES = os.environ.get('AXES', '0') == '1'

face_detector = detector.FaceDetector(MODEL, conf=0.5)
sticker_list = stickers.load(STICKER_DIR, PICK)


def process(frame):
    """프레임 하나를 처리한다. 얼굴을 잡았으면 True."""
    faces = face_detector.detect(frame)
    hit = False

    for face in faces:
        if not headpose.looks_valid(face):
            continue
        solved = headpose.solve(face, frame.shape)
        if solved is None:
            continue
        rvec, tvec, cam = solved

        compose.render(frame, sticker_list, rvec, tvec, cam)
        if SHOW_AXES:
            compose.draw_axes(frame, rvec, tvec, cam)
        hit = True

    return hit


if __name__ == '__main__':
    camera.run(process, TITLE)
