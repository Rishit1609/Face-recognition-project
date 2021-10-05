import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/elon musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloctest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255, 0, 255), 2)

faceDis = face_recognition.face_distance([encodeElon], encodeTest)
results = face_recognition.compare_faces([encodeElon], encodeTest)

print(results, faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}', (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
