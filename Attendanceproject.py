import cv2
import numpy as np
import face_recognition
import os

path = 'C:/Users/KIIT/PycharmProjects/Face Recognition/ImagesAttendance'
images = []
classname = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0])
print(classname)

def findEncodings(images):
    encodelist = []
    for img in images:
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findEncodings(images)
print('Encoding compelete')


cap = cv2.videoCapture(0)

while true:
    success, img = cap.read()
    imgS =cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgS)
    encodescurframe = face_recognition.face_encodings(imgS,facecurframe)