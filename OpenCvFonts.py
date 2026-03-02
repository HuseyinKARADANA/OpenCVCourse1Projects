import cv2
import numpy as np

screen=np.ones((800,800,3),np.uint8)*255
text="OpenCV"
color=(255,148,50)
for i in range(8):
    cv2.putText(screen,text,(80,80+i*80),i,4,color,2)
cv2.imshow("Deneme",screen)
cv2.waitKey(0)
cv2.destroyAllWindows()

