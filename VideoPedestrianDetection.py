import cv2
import imutils


cap=cv2.VideoCapture("files/yayalar.mp4")

while True:
    ret,frame=cap.read()
    if ret==False:
        break
    frame=imutils.resize(frame,1200)
    hog=cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cordinate,_=hog.detectMultiScale(frame,winStride=(4,4),
                                     padding=(8,8),
                                     scale=1.05)

    color=(0,0,255)
    thickness=5
    for (x,y,w,h) in cordinate:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,thickness)


    cv2.imshow("Original Img",frame)
    if cv2.waitKey(30) & 0XFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()