from flask import Flask, request, jsonify, Response
import cv2
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import numpy as np
import face_recognition
from datetime import datetime
import pandas as pd
import pickle

app = Flask(__name__)

recognizing = False
recognition_thread = None
cap = None

# Load training images and encode faces
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def load_encodings(file_path='encodings.pkl'):
    with open(file_path, 'rb') as f:
        encodings = pickle.load(f)
    print('Encodings loaded from', file_path)
    return encodings

encode_list_known = load_encodings('encoding_passport_size_large.pkl')
print('Encoding Complete')

def mark_attendance(name):
    file_path = 'Attendance.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time\n')
    
    with open(file_path, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list]
        
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'{name},{dt_string}\n')

def face_recognition_process():
    global cap
    cap = cv2.VideoCapture(0)
    while recognizing:
        success, img = cap.read()
        if not success:
            break

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        
        faces_cur_frame = face_recognition.face_locations(img_small)
        encodes_cur_frame = face_recognition.face_encodings(img_small, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = classNames[match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        _, jpeg = cv2.imencode('.jpg', img)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global recognizing, recognition_thread
    if not recognizing:
        recognizing = True
        recognition_thread = threading.Thread(target=face_recognition_process)
        recognition_thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already running'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global recognizing
    if recognizing:
        recognizing = False
        recognition_thread.join()  # Ensure the thread has finished
        
        # Send the email with the CSV file attached
        send_email(
            "Face Recognition Attendance",
            "The face recognition has been stopped. Please find the attached attendance CSV file.",
            "nj6604053@gmail.com",
            attachment_file='Attendance.csv'
        )
        
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not running'})

@app.route('/video_feed')
def video_feed():
    return Response(face_recognition_process(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def send_email(subject, body, to_email, attachment_file=None):
    try:
        from_email = "njytc001@gmail.com"
        password = "omqs mzve mikw cybg"

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if attachment_file:
            with open(attachment_file, 'rb') as f:
                part = MIMEApplication(f.read(), Name=attachment_file)
                part['Content-Disposition'] = f'attachment; filename="{attachment_file}"'
                msg.attach(part)

        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    app.run(debug=True)
