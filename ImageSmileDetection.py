import cv2

img=cv2.imread("files/smile.jpg")
face_cascade=cv2.CascadeClassifier("haarCascade/frontalface.xml")
smile_cascade=cv2.CascadeClassifier("haarCascade/smile.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3,3)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

face=img[x:x+w,y:y+h]
gray2=gray[x:x+w,y:y+h]

smiles=smile_cascade.detectMultiScale(gray2,1.3,5)
for (sx,sy,sw,sh) in smiles:
    cv2.rectangle(face,(sx,sy),(sx+sw,sy+sh),(0,255,0),3)



cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()