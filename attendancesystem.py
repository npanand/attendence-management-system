import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import mysql.connector

path = 'C:\\Users\\User\\Desktop\\cV\\photos'
images = []
classNames = []

myList = os.listdir(path)
#print(myList)

for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])
print("Class Name:")
#print(classNames)


def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList


def markAttendance(name):
	with open('attendancesystem.csv','r+') as f:
		myDataList = f.readlines()
#		print(myDataList)
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtstring = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtstring}')



encodeListKnown = findEncodings(images)
print("encoding completed")

cam = cv2.VideoCapture(0)

while True:
	success, img = cam.read()
	imgS = cv2.resize(img,(0,0),None,0.25,0.25)
	imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

	facelocCurframe = face_recognition.face_locations(imgS)
	encodeCurrentface = face_recognition.face_encodings(imgS,facelocCurframe)

	for encodeface, faceloc in zip(encodeCurrentface,facelocCurframe):
		matches = face_recognition.compare_faces(encodeListKnown,encodeface)
		faceDis = face_recognition.face_distance(encodeListKnown,encodeface)
		print(faceDis)
		matchIndex = np.argmin(faceDis)

		if matches[matchIndex]:
			name = classNames[matchIndex].upper()
			print(name)
			y1,x2,y2,x1 = faceloc
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
			cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,255),cv2.FILLED)
			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
			markAttendance(name)


	cv2.imshow('webcam',img)
	if cv2.waitKey(100) & 0xff == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
