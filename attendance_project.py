# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:18:27 2021

@author: VISHWESH
"""
#import tensorflow as tf
import dlib, cmake
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = "G:/ML 2.0/face recognition/images" #specify the path of images to be detected
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)    
#%%
def findEncodings(images):
    encodeList=[]
    for img in images:
        #img = cv2.resize(img, (540, 270),interpolation = cv2.INTER_NEAREST)
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # openncv used bgr color formatting, hence convert it to rgb
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f: #write a new csv file in the path
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now =datetime.now()
            dtsring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtsring}')
    
encodedListKnown = findEncodings(images)
print('Encoding complete!')



#%%

cap =cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    imgS = cv2.resize(img,(0,0),None,fx=0.25,fy=0.25) #img is reduced to 1/4 of its size
    imgS =cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodedListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodedListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)
                   
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()      
            
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    