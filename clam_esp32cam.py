import cv2
import numpy as np
import joblib
import json
import time
import pandas as pd 
from datetime import datetime
import os
import requests  # <--- NEW: Needed to fetch images from ESP32

# =================== ESP32 CAM SETTINGS ===================
ESP32_URL = "http://192.168.1.21/capture"  # <--- Your ESP32 IP address
# ==========================================================

# Load model and feature names
print("🔍 Loading model...")
model = joblib.load('clam_model_engineered.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
print(f"✅ Model loaded! Features: {feature_names}")

# ----------------------------------------------------------
# NEW FUNCTION: Get image from ESP32-CAM instead of Webcam
# ----------------------------------------------------------
def get_esp32_frame():
    try:
        response = requests.get(ESP32_URL, timeout=5)
        if response.status_code == 200:
            # Convert the downloaded bytes into a usable OpenCV image
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        print(f"⚠️ ESP32 Connection Error: {e}")
    return None
# ----------------------------------------------------------

def extract_features_from_frame(frame):
    """Extract exactly the same features as training (17 features)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 5000:
        return None
    
    # Calculate basic features
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
    
    # --- ENGINEERED FEATURES (same as training) ---
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
        'Blue_Green_Ratio': mean_blue / (mean_green + 1),
        'Green_Red_Ratio': mean_green / (mean_red + 1),
        'Blue_Minus_Green': mean_blue - mean_green,
        'Aspect_Circularity': aspect_ratio * circularity,
        'Shape_Score': aspect_ratio / (circularity + 0.01),
        'Total_Color_Variation': std_blue + std_green + std_red,
        'Color_Variation_Product': std_blue * std_green * std_red,
        'Mean_Green_Normalized': mean_green / (mean_blue + mean_green + mean_red)
    }
    
    # Return in correct order
    return [features[name] for name in feature_names]

def monitor_clam(check_interval=30, min_confidence=75):
    
    print(f"\n✅ ESP32-CAM Connected to {ESP32_URL}!")
    print(f"🔄 Checking every {check_interval} seconds")
    print(f"⚠️  DEAD alert threshold: {min_confidence}%")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    last_check = 0
    dead_count = 0
    alive_count = 0
    
    try:
        while True:
            current_time = time.time()
            if current_time - last_check >= check_interval:
                
                # 1. Get the frame from the ESP32-CAM
                frame = get_esp32_frame()
                
                if frame is not None:
                    # 2. Extract features
                    features = extract_features_from_frame(frame)
                    
                    if features is not None:
                        # DataFrame with column names
                        features_df = pd.DataFrame([features], columns=feature_names)
                        
                        # Predict using DataFrame
                        prediction = model.predict(features_df)[0]
                        prob = model.predict_proba(features_df)[0]
                        
                        status = "ALIVE" if prediction == 1 else "DEAD"
                        confidence = max(prob) * 100
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if status == "ALIVE":
                            alive_count += 1
                            print(f"[{timestamp}] ✅ ALIVE (Confidence: {confidence:.1f}%)")
                        else:
                            dead_count += 1
                            print(f"[{timestamp}] ⚠️ DEAD (Confidence: {confidence:.1f}%)")
                            
                            if confidence >= min_confidence:
                                print(f"   🔴 HIGH CONFIDENCE DEAD! (#{dead_count})")
                                # Save image
                                os.makedirs("dead_clams", exist_ok=True)
                                cv2.imwrite(f"dead_clams/dead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
                    else:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] ⚪ No clam features detected")
                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] ⚪ Failed to get image from ESP32")
                
                last_check = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    
    print(f"\n📊 Final Monitoring Summary of Pholas Orientalis:")
    print(f"   Alive Monitor Count: {alive_count}")
    print(f"   Dead Monitor Count: {dead_count}")

if __name__ == "__main__":
    print("🐚 Clam Health Monitor with ESP32-CAM")
    print("=" * 50)
    
    # Start monitoring
    monitor_clam(check_interval=30, min_confidence=75)