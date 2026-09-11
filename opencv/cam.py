import cv2 

cap = cv2.VideoCapture(0)

# 저장할 파일의 속성 정의하기
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

if cap.isOpened() :
    print("카메라가 잡혔어요!")
else :
    print("카메라가 안 잡혔어요ㅠㅠ")

while True :
    ret, frame = cap.read()

    if not ret : 
        break 

    out.write(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyWindow('frame')