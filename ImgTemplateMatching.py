import cv2
import numpy as np

image="files/starwars.jpg"
template="files/starwars2.jpg"

img=cv2.imread(image)
temp=cv2.imread(template,0)
w,h=temp.shape[::-1]
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



result=cv2.matchTemplate(gray_img,temp,cv2.TM_CCOEFF_NORMED)

location=np.where(result>=0.9)


for point in zip(*location[::-1]):
    print(point)

    cv2.rectangle(img,point,(point[0]+w,point[1]+h),(0,255,0),3)

cv2.imshow("img",img)
cv2.imshow("template",temp)
cv2.imshow("result",result)

cv2.waitKey(0)
cv2.destroyAllWindows()