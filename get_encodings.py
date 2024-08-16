import numpy as np
import cv2
import os
import pickle
from datetime import datetime
import face_recognition

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def save_encodings(encodings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(encodings, f)
    print('Encodings saved to', file_path)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, num_jitters=2, model='large')[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
save_encodings(encodeListKnown,'encodings.pkl')