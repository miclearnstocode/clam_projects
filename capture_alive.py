import cv2
import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, Response, jsonify

# =================== SETTINGS ===================
CAMERA_ID = 4
SAVE_DIR = "alive_detection"
PORT = 5001 # Using a different port than app.py so they don't conflict
# ================================================

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()
img_counter = 0

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
        return camera

def generate_frames():
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # HTML for the web interface
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>GoPro Capture</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
            h1 { color: #333; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            img { width: 100%; border-radius: 5px; margin-top: 10px; }
            button { background-color: #4CAF50; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 10px 2px; cursor: pointer; border: none; border-radius: 5px; }
            button:hover { background-color: #45a049; }
            .status { margin-top: 15px; font-weight: bold; color: #555; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📸 GoPro Clam Capture</h1>
            <p>Position the clams in the bucket, then click the button below.</p>
            <img src="/video_feed" alt="Live Feed">
            <br>
            <button onclick="captureImage()">CAPTURE IMAGE</button>
            <div class="status" id="status">Ready to capture...</div>
        </div>

        <script>
            function captureImage() {
                document.getElementById('status').innerText = 'Capturing...';
                fetch('/capture')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerText = '✅ Saved: ' + data.filename;
                    })
                    .catch(error => {
                        document.getElementById('status').innerText = '❌ Error capturing image';
                    });
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global img_counter
    cam = get_camera()
    success, frame = cam.read()
    
    if not success:
        return jsonify({"error": "Failed to grab frame"}), 500

    # Create directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alive_{timestamp}_{img_counter:04d}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    img_counter += 1
    print(f"📸 Captured: {filepath}")
    
    return jsonify({"filename": filename})

if __name__ == "__main__":
    print("=" * 50)
    print("🌐 GoPro Capture Server")
    print("=" * 50)
    print(f"1. Open your web browser (Chrome, Edge, etc.)")
    print(f"2. Go to: http://127.0.0.1:{PORT}")
    print(f"3. Click the 'CAPTURE IMAGE' button to save photos.")
    print(f"4. Images will be saved in the '{SAVE_DIR}' folder.")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)