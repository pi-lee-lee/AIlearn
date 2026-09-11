import cv2 
import numpy as np 

img = cv2.imread('opencv/lena.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 이후 자유롭게 후처리 가능!
h, w = img.shape[:2]
# 분류기 로드
det = cv2.FaceDetectorYN_create('opencv/yunet.onnx', '',(w,h))
ret, faces = det.detect(img)

# 검출 (입력 이미지는 그레이스케일 권장)
# scaleFactor: 이미지 피라미드 스케일 (보통 1.1)
# minNeighbors: 검출된 영역이 얼마나 중복되어야 얼굴로 인정할지 (보통 3~5)

for f in faces:
    x,y,w,h = f[:4].astype(int)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


cv2.imshow('fff', img)
cv2.waitKey()
cv2.destroyAllWindows()