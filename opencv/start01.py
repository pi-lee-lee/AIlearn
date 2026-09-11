# 허프 변환으로 직선 검출하는 예제 
import cv2
import numpy as np 

img = cv2.imread("opencv/shape.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 100)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
# Canny 에지 이미지에서 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                        minLineLength=100, maxLineGap=10)

cir = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT, 1.0, 10.0, param1=50.0, param2=20.0 , minRadius=20, maxRadius=200)

dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # 컬러로 변환해 그리기

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

if cir is not None:
    cir = np.uint16(np.around(cir))
    for c in cir[0]:
        cx, cy, r = c  # 중심좌표(cx, cy)와 반지름(r)
        
        # 초록색(0, 255, 0)으로 원 테두리 그리기
        cv2.circle(dst, (cx, cy), r, (0, 255, 0), 2)
        # 원의 중심점 점 찍기
        cv2.circle(dst, (cx, cy), 2, (0, 0, 255), 3)


cv2.imshow("img", img)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()