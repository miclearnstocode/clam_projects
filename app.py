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

# =================== GOPRO SETTINGS ===================
CAMERA_ID = 4  # <-- Your UVC GoPro
CHECK_INTERVAL = 5  # Seconds between AI checks
MIN_CONFIDENCE = 75  # Confidence threshold for DEAD alert
MODEL_PATH = 'clam_model_engineered.pkl'
FEATURES_PATH = 'feature_names.json'
DB_NAME = 'clam_monitor.db'
PREVIEW_DIR = 'preview'
# =====================================================

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()
latest_status = {
    "status": "Waiting...",
    "confidence": 0.0,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "image_path": None
}

# --- Initialize Database ---
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

# --- Load the Model ---
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

# --- Camera Management ---
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

# --- Feature Extraction ---
def extract_features_and_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 5000:
        return None, None
    
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(frame[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    features = {
        'Aspect_Ratio': aspect_ratio, 'Circularity': circularity, 'Solidity': solidity,
        'Mean_Blue': mean_blue, 'Mean_Green': mean_green, 'Mean_Red': mean_red,
        'Std_Blue': std_blue, 'Std_Green': std_green, 'Std_Red': std_red,
        'Blue_Green_Ratio': mean_blue / (mean_green + 1),
        'Green_Red_Ratio': mean_green / (mean_red + 1),
        'Blue_Minus_Green': mean_blue - mean_green,
        'Aspect_Circularity': aspect_ratio * circularity,
        'Shape_Score': aspect_ratio / (circularity + 0.01),
        'Total_Color_Variation': std_blue + std_green + std_red,
        'Color_Variation_Product': std_blue * std_green * std_red,
        'Mean_Green_Normalized': mean_green / (mean_blue + mean_green + mean_red)
    }
    return [features[name] for name in feature_names], main_contour

# --- Helper to map model prediction to status ---
def get_prediction_status(prediction):
    if prediction == 1:
        return "ALIVE"
    elif prediction == 0:
        return "DEAD"
    else:
        return "No Clam Detected"

# --- Background Monitoring Loop (AI Detection for DB) ---
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
            
            features, _ = extract_features_and_contour(frame)
            if features is not None:
                features_df = pd.DataFrame([features], columns=feature_names)
                prediction = model.predict(features_df)[0]
                prob = model.predict_proba(features_df)[0]
                status = get_prediction_status(prediction)
                confidence = max(prob) * 100
                details = "Standard detection"
                
                image_path = None
                # Only save image if it's a high-confidence DEAD detection
                if status == "DEAD" and confidence >= MIN_CONFIDENCE:
                    image_path = f"dead_clams/dead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(image_path, frame)
                    details = "HIGH CONFIDENCE DEAD! Image saved."
                    print(f"🔴 {details}")
                elif status == "DEAD" and confidence < MIN_CONFIDENCE:
                    # If low confidence on dead, consider it as no clam or hiding
                    status = "No Clam Detected"
                    confidence = 0.0
                    details = "Low confidence on dead (likely hiding)"
                    print(f"⚪ {details}")
                
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

# --- MJPEG Stream Generator (With Live Overlay) ---
def generate_frames():
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            release_camera()
            time.sleep(1)
            continue
        
        # DRAW DETECTION OVERLAY
        features, contour = extract_features_and_contour(frame)
        
        if features is not None:
            features_df = pd.DataFrame([features], columns=feature_names)
            prediction = model.predict(features_df)[0]
            prob = model.predict_proba(features_df)[0]
            status = get_prediction_status(prediction)
            confidence = max(prob) * 100

            # Color-coded box
            if status == "ALIVE":
                box_color = (0, 255, 0)  # Green
            elif status == "DEAD":
                box_color = (0, 0, 255)  # Red
            else:
                box_color = (255, 255, 0) # Cyan for No Clam

            # Draw the Contour
            cv2.drawContours(frame, [contour], -1, box_color, 2)

            # Draw Bounding Box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw Status Text Box
            text = f"{status} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x, y - 35), (x + 320, y), box_color, cv2.FILLED)
            cv2.putText(frame, text, (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Encode frame
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

# Fallback for dead_clams
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