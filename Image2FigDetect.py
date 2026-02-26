import cv2

img=cv2.imread("files/fig.jpg")
img=cv2.resize(img,(1000,1000))
driedfig_cascade=cv2.CascadeClassifier("haarCascade/dried_fig_cascade.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

figs=driedfig_cascade.detectMultiScale(gray,1.1,3)

for (x,y,w,h) in figs:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("Dried Fig",img)
cv2.waitKey(0)
cv2.destroyAllWindows()