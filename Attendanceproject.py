import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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

def markattendance(name):
    with open('Attendance.csv','r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodelistknown = findEncodings(images)
print('Encoding compelete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS =cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgS)
    encodescurframe = face_recognition.face_encodings(imgS,facecurframe)

    for encodeface,faceloc in zip(encodescurframe,facecurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown, encodeface)
        #print(facedis)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = classname[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,y2,x1,x2 = y1*4,y2*4,x1*4,x2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            markattendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
