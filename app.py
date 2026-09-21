import cv2
import os
import time
import threading
import sqlite3
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, Response, abort
from ultralytics import YOLO

# =================== SETTINGS ===================
CAMERA_ID = 4                
CHECK_INTERVAL = 5            
YOLO_MODEL_PATH = 'pholas_yolo.pt'
DB_NAME = 'clam_monitor.db'
PREVIEW_DIR = 'preview'

# YOLO confidence threshold for a detection to count
CONF_THRESHOLD = 0.40

# Map dataset class names -> your app's status labels
# Roboflow: 0='At risk', 1='Healthy'
# If you find labels are inverted on live view, swap these two values.
CLASS_TO_STATUS = {
    'Healthy':  'ALIVE',
    'At risk':  'DEAD',
}
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

model = None   # YOLO model


# ---------------- DB ----------------
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
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          status, confidence, details, image_path))
    conn.commit()
    conn.close()


# ---------------- Model ----------------
def load_model():
    global model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
        print(f"   Classes: {model.names}")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        exit()


# ---------------- Camera ----------------
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


# ---------------- Inference ----------------
def detect_clams(frame):
    """
    Run YOLOv8 on a frame. Returns list of:
       (label, confidence_float, (x1, y1, x2, y2))
    """
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        verbose=False,
        device="cpu",          # Intel Iris Xe -> CPU
        imgsz=480,             # matches training size (fast on CPU)
    )

    detections = []
    if not results:
        return detections

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return detections

    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((label, conf, (x1, y1, x2, y2)))
    return detections


# ---------------- Monitor loop ----------------
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

            detections = detect_clams(frame)

            if detections:
                statuses = []
                confidences = []

                for label, conf, _ in detections:
                    status = CLASS_TO_STATUS.get(label, "UNKNOWN")
                    statuses.append(status)
                    confidences.append(conf * 100)

                # Tank-level rule: if ANY clam is ALIVE, tank is ALIVE
                status = "ALIVE" if "ALIVE" in statuses else "DEAD"
                confidence = max(confidences)
                details = f"Detected {len(detections)} clams"

                image_path = None
                if status == "DEAD":
                    image_path = (
                        f"dead_clams/dead_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )
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
                save_to_db("No Clam Detected", 0.0, "YOLO found nothing", preview_path)
                latest_status = {
                    "status": "No Clam Detected",
                    "confidence": 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": preview_path
                }
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No clam detected")
        else:
            release_camera()
            save_to_db("GoPro Error", 0.0, "Failed to fetch image from GoPro", None)
            latest_status = {
                "status": "GoPro Error",
                "confidence": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to get image")

        time.sleep(CHECK_INTERVAL)


# ---------------- Live stream ----------------
def generate_frames():
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            release_camera()
            time.sleep(1)
            continue

        detections = detect_clams(frame)

        for label, conf, (x1, y1, x2, y2) in detections:
            status = CLASS_TO_STATUS.get(label, "UNKNOWN")
            color = (0, 255, 0) if status == "ALIVE" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{status} ({conf*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + 300, y1), color, cv2.FILLED)
            cv2.putText(frame, text, (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ---------------- Flask Routes ----------------
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
    c.execute('''SELECT timestamp, status, confidence, details, image_path
                 FROM clam_logs ORDER BY id DESC LIMIT 50''')
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)


@app.route('/<path:filename>')
def serve_static_files(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    abort(404)


# ---------------- Main ----------------
if __name__ == "__main__":
    init_db()
    load_model()

    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    print("🔄 Background monitoring started...")

    print("🌐 Dashboard: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)