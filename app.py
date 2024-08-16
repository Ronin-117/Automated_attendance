from flask import Flask, render_template, Response, send_file, jsonify
import pandas as pd
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

app = Flask(__name__)

# Load known face encodings and names
def load_encodings(file_path='encoding_passport_size_large.pkl'):
    with open(file_path, 'rb') as f:
        encodings = pickle.load(f)
    print('Encodings loaded from', file_path)
    return encodings

def markAttendance(name):
    file_path = 'Attendance.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time\n')
    with open(file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')

# Load face encodings and names
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

loaded_encodings = load_encodings('encoding_passport_size_large.pkl')

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        else:
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS, model='hog')
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame, num_jitters=2, model='large')

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(loaded_encodings, encodeFace, tolerance=0.5)
                faceDis = face_recognition.face_distance(loaded_encodings, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                continue
            frame = buffer.tobytes()
            # Yield the frame in the correct format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    print("Index route accessed.")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("Video feed route accessed.")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_csv')
def get_csv():
    # Path to your CSV file
    file_path = 'Attendance.csv'
    return send_file(file_path, mimetype='text/csv', as_attachment=False)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
