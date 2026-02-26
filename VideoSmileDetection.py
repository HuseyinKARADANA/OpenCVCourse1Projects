import cv2

vid=cv2.VideoCapture("files/smile.mp4")
#vid=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarCascade/frontalface.xml")
smile_cascade=cv2.CascadeClassifier("haarCascade/smile.xml")

while True:
    ret,frame=vid.read()
    frame=cv2.resize(frame,(600,360))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.5,9)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        roi_frame=frame[x:x+w,y:y+h]
        roi_gray=gray[x:x+w,y:y+h]

        smiles=smile_cascade.detectMultiScale(roi_gray,1.5,9)

        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_frame,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

    cv2.imshow("video",frame)

    if cv2.waitKey(20) & 0xFF== ord("q"):
        break
vid.release()
cv2.destroyAllWindows()