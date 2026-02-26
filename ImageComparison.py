import cv2
import numpy as np

path="files/aircraft.jpg"
path2="files/aircraft_copy.jpg"
img1= cv2.imread(path)
img2=cv2.imread(path)
#img1=cv2.resize(img1,(640,550))
img3=cv2.medianBlur(img1,7)

if img1.shape==img2.shape:
    print("Same Size")
else:
    print("Not Same")

#diff=difference anlammında yazılır
diff=cv2.subtract(img1,img3)

b,g,r = cv2.split(diff)


if cv2.countNonZero(b)==0 and cv2.countNonZero(g)==0 and cv2.countNonZero(r)==0:
    print("completely equal")
else:
    print("Not completely equal")

cv2.imshow("Aircraft 1",img1)
cv2.imshow("Aircraft 2",img2)
cv2.imshow("Aircraft 3",img3)
cv2.imshow("Difference",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
