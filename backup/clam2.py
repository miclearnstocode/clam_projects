import cv2
import numpy as np
import joblib
import json
import time
from datetime import datetime

# Load your new model
model = joblib.load('clam_model_no_overfit.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

def extract_features(frame):
    """Extract features from frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 5000:
        return None
    
    # Basic features
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Color features
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(frame[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    features = [aspect_ratio, circularity, solidity, 
                mean_blue, mean_green, mean_red, 
                std_blue, std_green, std_red]
    
    return features

def monitor_clams(camera_id=2, interval=30, dead_threshold=75):
    """Monitor clams with optimized model"""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("🐚 Clam Health Monitor - OPTIMIZED MODEL")
    print(f"✅ Model: {model.oob_score_*100:.1f}% OOB score")
    print(f"📷 Camera: {camera_id}")
    print(f"⏱️  Check interval: {interval}s")
    print(f"⚠️  Dead threshold: {dead_threshold}%")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    last_check = 0
    dead_count = 0
    alive_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current = time.time()
            if current - last_check >= interval:
                features = extract_features(frame)
                
                if features:
                    pred = model.predict([features])[0]
                    prob = model.predict_proba([features])[0]
                    
                    status = "ALIVE" if pred == 1 else "DEAD"
                    confidence = max(prob) * 100
                    dead_conf = prob[0] * 100
                    alive_conf = prob[1] * 100
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    if status == "ALIVE":
                        alive_count += 1
                        print(f"[{timestamp}] ✅ ALIVE (Conf: {alive_conf:.1f}%)")
                    else:
                        dead_count += 1
                        if dead_conf >= dead_threshold:
                            print(f"[{timestamp}] ⚠️ DEAD (Conf: {dead_conf:.1f}%)")
                            # Save image for verification
                            cv2.imwrite(f"dead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
                        else:
                            print(f"[{timestamp}] 🟡 DEAD (Low conf: {dead_conf:.1f}%)")
                    
                    # Show summary every 10 checks
                    if (alive_count + dead_count) % 10 == 0:
                        print(f"📊 Summary: Alive={alive_count}, Dead={dead_count}")
                        
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚪ No clam detected")
                
                last_check = current
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")
    
    cap.release()
    print(f"\n📊 Final Summary:")
    print(f"   Alive detected: {alive_count}")
    print(f"   Dead detected: {dead_count}")

# Run monitoring
if __name__ == "__main__":
    monitor_clams(camera_id=2, interval=30, dead_threshold=75)