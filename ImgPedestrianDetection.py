import cv2
import imutils

img=cv2.imread("files/yaya.jpg")
img=imutils.resize(img,750)

hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cordinate,_=hog.detectMultiScale(img,winStride=(4,4),
                                 padding=(8,8),
                                 scale=1.05)

color=(0,0,255)
thickness=5
for (x,y,w,h) in cordinate:
    cv2.rectangle(img,(x,y),(x+w,y+h),color,thickness)











cv2.imshow("Original Img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()