import cv2
import numpy as np


img = cv2.imread('opencv/coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("dd",blurred)

_, thresh =  cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
# kernel = np.ones((7, 7), np.uint8)

# # 침식
# erosion = cv2.erode(thresh, kernel, iterations=1)
# # 팽창
# dialtion = cv2.dilate(thresh, kernel, iterations=1)

# 열림 후 추가 팽창
# result1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# result1 = cv2.dilate(result1, kernel, iterations=1)
blurred = cv2.GaussianBlur(thresh, (7, 7), 0)


cir = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT, dp=1.0, minDist=10.0, param1=100.0, param2=40.0 , minRadius=40, maxRadius=70)
count = 0
if cir is not None:
    cir = np.uint16(np.around(cir))
    count = cir.size//3

contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

text = f"found {count} coins"

cv2.putText(
    img,
    text,
    (10,40),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1.2,
    color=(0, 0, 0),  # 초록색 (B, G, R)
    thickness=2
)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imshow("dd",img)
cv2.waitKey(0)