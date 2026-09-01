import cv2
import numpy as np
import joblib
import json
import os
import time
import threading
import sqlite3
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, Response, abort

# =================== SETTINGS ===================
CAMERA_ID = 0
CHECK_INTERVAL = 5
MODEL_PATH = 'clam_model_engineered.pkl'
FEATURES_PATH = 'feature_names.json'
DB_NAME = 'clam_monitor.db'
PREVIEW_DIR = 'preview'
# ================================================

app = Flask(__name__)

camera = None
camera_lock = threading.Lock()
latest_status = {
    "status": "Waiting...",
    "confidence": 0.0,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "image_path": None
}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS clam_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            confidence REAL,
            details TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(status, confidence, details, image_path=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO clam_logs (timestamp, status, confidence, details, image_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, confidence, details, image_path))
    conn.commit()
    conn.close()

def load_model():
    global model, feature_names
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'r') as f:
            feature_names = json.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def detect_multiple_clams(frame):
    h, w, _ = frame.shape

    # Only look at the BOTTOM 40% of the frame
    start_y = int(h * 0.60)
    cropped = frame[start_y:, :]  
    cropped_h, cropped_w, _ = cropped.shape

    # Convert to HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    
    # Look for the clam shells (Pale/White with texture)
    lower_clam = np.array([0, 20, 80])
    upper_clam = np.array([180, 100, 255])
    
    mask = cv2.inRange(hsv, lower_clam, upper_clam)

    # Merge the clams
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_clams = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Large area filter
        if area < 30000 or area > 300000:
            continue

        # Reject perfect circles (bubbles)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.70:
            continue

        # Relaxed Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.2 or aspect_ratio > 6.0:
            continue

        # Add crop offset
        y += start_y

        # Feature extraction
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity > 0.98:
            continue

        mask_bgr = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask_bgr, [contour], -1, 255, -1)
        mean_color = cv2.mean(cropped, mask=mask_bgr)[:3]
        mean_blue, mean_green, mean_red = mean_color

        std_dev = np.std(cropped[mask_bgr == 255], axis=0)
        std_blue, std_green, std_red = std_dev

        # === PINK/WHITE NOSE DETECTION ===
        lower_pink = np.array([130, 80, 80])
        upper_pink = np.array([180, 255, 255])
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        pink_in_contour = cv2.countNonZero(cv2.bitwise_and(mask_pink, mask_bgr))
        pink_ratio = pink_in_contour / area if area > 0 else 0

        # White mask (Dead indicator)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        white_in_contour = cv2.countNonZero(cv2.bitwise_and(mask_white, mask_bgr))
        white_ratio = white_in_contour / area if area > 0 else 0

        # Redness and Pink Intensity
        redness = mean_red - mean_green
        pink_intensity = mean_red / (mean_blue + 1)
        
        # Pink Dominance
        pink_dominance = (pink_ratio * 100) - (white_ratio * 10)

        features = {
            'Aspect_Ratio': aspect_ratio, 
            'Circularity': circularity, 
            'Solidity': solidity,
            'Mean_Blue': mean_blue, 
            'Mean_Green': mean_green, 
            'Mean_Red': mean_red,
            'Std_Blue': std_blue, 
            'Std_Green': std_green, 
            'Std_Red': std_red,
            'Pink_Ratio': pink_ratio,
            'White_Ratio': white_ratio,
            'Redness': redness,
            'Pink_Intensity': pink_intensity,
            'Pink_Dominance': pink_dominance
        }
        
        feature_vector = [features[name] for name in feature_names]
        detected_clams.append((feature_vector, contour, (x, y, w, h)))

    # Limit to largest 3 clams
    detected_clams.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
    return detected_clams[:3]

def get_prediction_status(prediction):
    # REVERSED LOGIC: If model says 0 (Dead) -> Show ALIVE
    # If model says 1 (Alive) -> Show DEAD
    if prediction == 0:
        return "ALIVE"
    else:
        return "DEAD"

def run_monitor():
    global latest_status
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs("dead_clams", exist_ok=True)

    while True:
        cam = get_camera()
        success, frame = cam.read()

        if success:
            preview_path = f"{PREVIEW_DIR}/latest.jpg"
            cv2.imwrite(preview_path, frame)

            detected_clams = detect_multiple_clams(frame)

            if detected_clams:
                all_statuses = []
                all_confidences = []

                for features, _, _ in detected_clams:
                    features_df = pd.DataFrame([features], columns=feature_names)
                    prediction = model.predict(features_df)[0]
                    prob = model.predict_proba(features_df)[0]
                    
                    # REVERSED LOGIC: Invert the prediction!
                    current_status = "ALIVE" if prediction == 0 else "DEAD"
                    confidence = max(prob) * 100

                    all_statuses.append(current_status)
                    all_confidences.append(confidence)

                # If ANY clam is ALIVE, the whole tank is ALIVE!
                if "ALIVE" in all_statuses:
                    status = "ALIVE"
                else:
                    status = "DEAD"
                
                confidence = max(all_confidences)
                details = f"Detected {len(detected_clams)} clams"

                image_path = None
                if status == "DEAD":
                    image_path = f"dead_clams/dead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(image_path, frame)
                    details = "DEAD DETECTED! Image saved."
                    print(f"🔴 {details}")

                save_to_db(status, confidence, details, image_path)
                latest_status = {
                    "status": status,
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": preview_path
                }
                print(f"[{latest_status['timestamp']}] {status} ({confidence:.1f}%)")
            else:
                save_to_db("No Clam Detected", 0.0, "No features extracted", preview_path)
                latest_status = {"status": "No Clam Detected", "confidence": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "image_path": preview_path}
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No clam detected")
        else:
            release_camera()
            save_to_db("GoPro Error", 0.0, "Failed to fetch image from GoPro", None)
            latest_status = {"status": "GoPro Error", "confidence": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to get image from GoPro")

        time.sleep(CHECK_INTERVAL)

def generate_frames():
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            release_camera()
            time.sleep(1)
            continue

        detected_clams = detect_multiple_clams(frame)
        frame_height = frame.shape[0]

        for features, contour, bbox in detected_clams:
            features_df = pd.DataFrame([features], columns=feature_names)
            prediction = model.predict(features_df)[0]
            prob = model.predict_proba(features_df)[0]
            
            # REVERSED LOGIC: Invert the prediction!
            if prediction == 0:
                status = "ALIVE"
                box_color = (0, 255, 0)  # Green
            else:
                status = "DEAD"
                box_color = (0, 0, 255)  # Red

            confidence = max(prob) * 100

            contour_offset = contour + np.array([0, int(frame_height * 0.60)])
            cv2.drawContours(frame, [contour_offset], -1, box_color, 2)

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            text = f"{status} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x, y - 30), (x + 280, y), box_color, cv2.FILLED)
            cv2.putText(frame, text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify(latest_status)

@app.route('/api/logs')
def api_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT timestamp, status, confidence, details, image_path FROM clam_logs ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

@app.route('/<path:filename>')
def serve_static_files(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    abort(404)

# --- Main Execution ---
if __name__ == "__main__":
    init_db()
    load_model()

    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    print("🔄 Background monitoring started...")

    print("🌐 Starting Web Dashboard with Live Stream at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)