import cv2
import numpy as np


img = cv2.imread('opencv/book.jpeg')
dst_img = np.zeros((480, 557, 3), dtype=np.uint8)

s_p = np.array([[60,28], [418, 19], [15,530],[458,530]] , dtype=np.float32)

d_p = np.array([[15,19],[458,19],[15,530],[458,530]], dtype=np.float32)

m  = cv2.getPerspectiveTransform(s_p, d_p)

r_i = cv2.warpPerspective(img,m,(480,557))


cv2.imshow("dd",r_i)

cv2.waitKey(0)