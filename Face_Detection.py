# To Detect Faces We are going to use a method proposed by VIOLA & JONES
# This was one of the earliest methods that allowed Real Time Face Detection
# Here we are not going to train a model we will use a pretrained file for faces which is provided by open CV
# openCV has provided some default cascades that can detect different things like Number Plates,Eyes,Full Body Etc
#..................../................./........................./..................../................./........

import cv2

faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

img = cv2.imread('Resources/Lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the faces in this image using our face cascade

faces = faceCascade.detectMultiScale(imgGray,1.1,4)
# Creating Bounding Box around faces we have detected
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+h,y+h),(255,0,0),2)


cv2.imshow("Result",img)
cv2.waitKey(0)