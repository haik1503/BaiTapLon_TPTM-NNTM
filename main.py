import cv2
import face_recognition
import os
import sqlite3
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
import random
import base64
import numpy as np

# ========== CẤU HÌNH ==========
KNOWN_FACE_DIR = 'known_faces'
DB_PATH = 'history.db'
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

# Khởi tạo Flask app
app = Flask(__name__)

# ========== CƠ SỞ DỮ LIỆU ==========
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    gender TEXT,
    timestamp TEXT
)''')
conn.commit()

# ========== TIỆN ÍCH ==========
def log_to_db(age, gender):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO visits (age, gender, timestamp) VALUES (?, ?, ?)", (age, gender, timestamp))
    conn.commit()

def estimate_age_gender(face_img):
    age = random.randint(5, 80)
    gender = random.choice(['Man', 'Woman'])
    return age, gender

def recognize_face(face_encoding):
    for filename in os.listdir(KNOWN_FACE_DIR):
        path = os.path.join(KNOWN_FACE_DIR, filename)
        known_img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(known_img)
        if not encs:
            continue
        match = face_recognition.compare_faces([encs[0]], face_encoding)
        if match[0]:
            return filename.split('.')[0]
    return None

def save_new_customer_image(face_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"customer_{timestamp}.jpg"
    path = os.path.join(KNOWN_FACE_DIR, filename)
    cv2.imwrite(path, face_img)
    return filename.split('.')[0]

def get_recommendations_with_location(age, gender):
    product_db = {
        'Bánh snack': {'shelf': 3, 'floor': 1},
        'Đồ chơi': {'shelf': 4, 'floor': 1},
        'Gấu bông': {'shelf': 5, 'floor': 1},
        'Truyện tranh': {'shelf': 6, 'floor': 1},
        'Mỹ phẩm': {'shelf': 2, 'floor': 2},
        'Nước hoa': {'shelf': 3, 'floor': 2},
        'Dao cạo râu': {'shelf': 2, 'floor': 1},
        'Dụng cụ thể thao': {'shelf': 5, 'floor': 1},
        'Áo sơ mi nam': {'shelf': 4, 'floor': 2},
        'Tạp chí sức khỏe': {'shelf': 6, 'floor': 2},
        'Rượu vang': {'shelf': 1, 'floor': 2},
        'Thực phẩm chức năng': {'shelf': 1, 'floor': 1},
        'Khăn quàng cổ': {'shelf': 3, 'floor': 1},
        'Kem dưỡng da': {'shelf': 2, 'floor': 2}
    }

    suggestions = []
    if gender == 'Man':
        if age < 12:
            suggestions = ['Bánh snack', 'Đồ chơi', 'Truyện tranh']
        elif age < 30:
            suggestions = ['Dao cạo râu', 'Dụng cụ thể thao', 'Áo sơ mi nam']
        elif age < 60:
            suggestions = ['Tạp chí sức khỏe', 'Rượu vang', 'Thực phẩm chức năng']
        else:
            suggestions = ['Thực phẩm chức năng', 'Tạp chí sức khỏe', 'Khăn quàng cổ']
    elif gender == 'Woman':
        if age < 12:
            suggestions = ['Bánh snack', 'Gấu bông', 'Truyện tranh']
        elif age < 30:
            suggestions = ['Mỹ phẩm', 'Nước hoa', 'Kem dưỡng da']
        elif age < 60:
            suggestions = ['Tạp chí sức khỏe', 'Nước hoa', 'Thực phẩm chức năng']
        else:
            suggestions = ['Thực phẩm chức năng', 'Khăn quàng cổ', 'Tạp chí sức khỏe']

    results = []
    for item in suggestions:
        info = product_db.get(item, {'shelf': 'N/A', 'floor': 'N/A'})
        results.append({'product': item, 'shelf': info['shelf'], 'floor': info['floor']})
    return results

# ========== XỬ LÝ VIDEO STREAM ==========
cap = cv2.VideoCapture(0)
current_customers = set()
known_customers = set()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        new_current = set()
        recommendations = []
        customer_info = None

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = recognize_face(face_encoding)
            top, right, bottom, left = [v * 4 for v in face_location]
            face_img = frame[top:bottom, left:right]

            if name:
                new_current.add(name)
                display_name = f"Cũ - {name}" if name in known_customers else "Mới"
                known_customers.add(name)
            else:
                name = save_new_customer_image(face_img)
                new_current.add(name)
                display_name = "Mới"

            age, gender = estimate_age_gender(face_img)
            log_to_db(age, gender)
            recommendations = get_recommendations_with_location(age, gender)
            customer_info = {'name': display_name, 'age': age, 'gender': gender}

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        current_customers.clear()
        current_customers.update(new_current)

        # Chuyển frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Lưu thông tin để trả về qua API
        global last_customer_info, last_recommendations
        last_customer_info = customer_info
        last_recommendations = recommendations

# ========== ROUTES FLASK ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recommendations')
def get_recommendations():
    global last_customer_info, last_recommendations
    return jsonify({
        'customer': last_customer_info if 'last_customer_info' in globals() else None,
        'recommendations': last_recommendations if 'last_recommendations' in globals() else []
    })

# ========== MAIN ==========
if __name__ == '__main__':
    app.run(debug=True)