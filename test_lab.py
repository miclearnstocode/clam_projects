import cv2
import numpy as np
import joblib
import json
import os
import time
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, Response

# ================= SETTINGS =================
MODEL_PATH = 'clam_model_engineered.pkl'
FEATURES_PATH = 'feature_names.json'
# ============================================

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

# Load model and features
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
except:
    print("❌ Error loading model or features! Make sure they exist.")
    exit()

# ==========================================================
# THE PINK ENHANCER (Uses Roberts Cross Edges + Saturation)
# ==========================================================
def apply_pink_enhancement(frame):
    # 1. Saturation +100
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 100, 0, 255)
    frame_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Contrast +100
    frame_sat = cv2.convertScaleAbs(frame_sat, alpha=2.0, beta=0)

    # 3. Roberts Cross Edge Detection
    gray = cv2.cvtColor(frame_sat, cv2.COLOR_BGR2GRAY)
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    edge_x = cv2.filter2D(gray, -1, roberts_x)
    edge_y = cv2.filter2D(gray, -1, roberts_y)
    edges = np.sqrt(edge_x**2 + edge_y**2)
    edges = np.uint8(np.clip(edges, 0, 255))

    # 4. Edge threshold (Roberts Cross = 100)
    _, edges_thresh = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # 5. Edge Enhancement 5x5
    kernel = np.ones((5, 5), np.float32) / 25
    edges_smooth = cv2.filter2D(edges_thresh, -1, kernel)

    # 6. Fuse edges with saturated image
    edges_bgr = cv2.cvtColor(edges_smooth, cv2.COLOR_GRAY2BGR)
    fused_frame = cv2.addWeighted(frame_sat, 0.8, edges_bgr, 0.2, 0)

    # 7. Find the PINK/Violet siphon using the ENHANCED image
    hsv = cv2.cvtColor(fused_frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (180, 255, 255))

    return fused_frame, pink_mask

def detect_multiple_clams(frame):
    h, w, _ = frame.shape
    cropped = frame.copy()

    # 1. Use the RAW image (No enhancement!)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    
    # 2. Find the PINK/SKIN TONE nose (RAW colors!)
    # The nose is in the RAW image: PINK (Hue 0-10), ORANGE (Hue 10-25)
    alive_mask = cv2.inRange(hsv, (0, 30, 100), (20, 255, 255))  # Skin/Pink/Orange
    dead_mask = cv2.inRange(hsv, (0, 0, 180), (180, 20, 255))    # White/Translucent (dead)

    # 3. Morphology to connect the tube
    kernel = np.ones((3, 3), np.uint8)
    alive_mask = cv2.morphologyEx(alive_mask, cv2.MORPH_CLOSE, kernel)
    dead_mask = cv2.morphologyEx(dead_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Find ALL contours
    alive_contours, _ = cv2.findContours(alive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dead_contours, _ = cv2.findContours(dead_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_clams = []

    # 5. Process ALIVE (Pink Nose)
    for contour in alive_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Must be a REAL tube (Not tiny background)
        if area < 3000 or area > 20000:
            continue

        # Must be long/thin
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue

        detected_clams.append((contour, (x, y, w, h), "ALIVE"))

    # 6. Process DEAD (White Nose)
    for contour in dead_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Must be a REAL white tube
        if area < 3000 or area > 20000:
            continue

        # Must be long/thin
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue

        # Check if there is PINK inside (If yes, it's actually alive)
        pink_inside = np.sum(alive_mask[y:y+h, x:x+w])
        if pink_inside > 100:
            continue

        detected_clams.append((contour, (x, y, w, h), "DEAD"))

    detected_clams.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
    return detected_clams[:10]

def overlay_results(frame, detected_clams):
    # Return to NORMAL image (not enhanced!)
    enhanced_frame = frame.copy()

    for contour, bbox, detected_status in detected_clams:
        x, y, w, h = bbox

        if detected_status == "ALIVE":
            status = "ALIVE"
            confidence = 100.0
            color = (0, 255, 0)  # Green
        else:
            status = "DEAD"
            confidence = 95.0
            color = (0, 0, 255)  # Red

        cv2.rectangle(enhanced_frame, (x, y), (x + w, y + h), color, 3)

        # Draw text label
        text = f"{status} ({confidence:.1f}%)"
        cv2.rectangle(enhanced_frame, (x, y - 30), (x + 230, y), color, cv2.FILLED)
        cv2.putText(enhanced_frame, text, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return enhanced_frame
# --- FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('test_lab.html')

@app.route('/test_image', methods=['POST'])
def test_image():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filename = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    os.makedirs('uploads', exist_ok=True)
    file.save(filename)
    
    frame = cv2.imread(filename)
    detected = detect_multiple_clams(frame)
    result_frame = overlay_results(frame, detected)
    result_path = f"uploads/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(result_path, result_frame)
    
    return render_template('test_lab.html', image_result=result_path)

@app.route('/test_video', methods=['POST'])
def test_video():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filename = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    os.makedirs('uploads', exist_ok=True)
    file.save(filename)
    
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    result_path = f"uploads/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected = detect_multiple_clams(frame)
        result_frame = overlay_results(frame, detected)
        out.write(result_frame)
    
    cap.release()
    out.release()
    
    return render_template('test_lab.html', video_result=result_path)

@app.route('/webcam_feed')
def webcam_feed():
    def generate():
        cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detected = detect_multiple_clams(frame)
            result_frame = overlay_results(frame, detected)
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    print("🐚 Clam Test Lab Running at http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)