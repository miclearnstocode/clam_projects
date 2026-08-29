import cv2
import numpy as np
import joblib
import json
import time
from datetime import datetime
import os
import pandas as pd

# ============================================
# LOAD MODEL AND FEATURES
# ============================================
print("🔍 Loading model...")
model = joblib.load('clam_model_no_overfit.pkl')

# Load feature names from JSON
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
print(f"✅ Model loaded! Features: {feature_names}")

# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_features_from_frame(frame):
    """Extract exactly the same features as training"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find largest contour
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 5000:
        return None, None
    
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
    
    # Return features in the order expected by the model
    features = [aspect_ratio, circularity, solidity, 
                mean_blue, mean_green, mean_red, 
                std_blue, std_green, std_red]
    
    return features, (x, y, w, h)

# ============================================
# PREDICT WITH FEATURE NAMES
# ============================================
def predict_with_features(model, features, feature_names):
    """Predict without warnings using feature names"""
    features_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    return prediction, probabilities

# ============================================
# CAPTURE AND AUTO-LABEL FUNCTION
# ============================================
def capture_and_auto_label(frame, model, feature_names, capture_count):
    """Capture frame and automatically label as alive or dead"""
    
    # Extract features
    features, bbox = extract_features_from_frame(frame)
    
    if features is None:
        print("❌ No clam detected in frame!")
        return capture_count, None, None, None
    
    # Predict
    prediction, probabilities = predict_with_features(model, features, feature_names)
    
    status = "ALIVE" if prediction == 1 else "DEAD"
    dead_conf = probabilities[0] * 100
    alive_conf = probabilities[1] * 100
    confidence = max(dead_conf, alive_conf)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine label and folder
    if status == "ALIVE":
        folder = "alive"
        label = "alive"
        emoji = "✅"
    else:
        folder = "dead"
        label = "dead"
        emoji = "⚠️"
    
    # Save to appropriate folder
    filename = f"test_captures/{folder}/{label}_{timestamp}_{confidence:.0f}%.jpg"
    cv2.imwrite(filename, frame)
    
    print(f"\n📸 {emoji} AUTO-CAPTURED: {status}")
    print(f"   File: {filename}")
    print(f"   DEAD confidence: {dead_conf:.2f}%")
    print(f"   ALIVE confidence: {alive_conf:.2f}%")
    print(f"   Confidence: {confidence:.2f}%")
    
    # Save metadata
    metadata_file = f"test_captures/{folder}/{label}_{timestamp}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Label: {label}\n")
        f.write(f"Prediction: {status}\n")
        f.write(f"DEAD confidence: {dead_conf:.2f}%\n")
        f.write(f"ALIVE confidence: {alive_conf:.2f}%\n")
        f.write(f"Features: {features}\n")
    
    return capture_count + 1, status, confidence, features

# ============================================
# TEST FUNCTION (NO GUI)
# ============================================
def test_model_with_webcam(camera_id=2):
    """Test the model with live webcam feed - NO GUI"""
    
    # Open camera
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}!")
        return
    
    print(f"\n📷 Iriun Webcam (Camera {camera_id}) opened!")
    print("=" * 60)
    print("🔬 TEST MODE - Auto-label captures")
    print("  Press keys in terminal to control:")
    print("  't' - Test current frame (predict alive/dead)")
    print("  'c' - Capture and auto-label (alive/dead)")
    print("  's' - Save current frame as test image (no label)")
    print("  'a' - Auto-capture every 10 seconds")
    print("  'q' - Quit")
    print("=" * 60)
    
    # Create folders for saved data
    os.makedirs("test_captures", exist_ok=True)
    os.makedirs("test_captures/alive", exist_ok=True)
    os.makedirs("test_captures/dead", exist_ok=True)
    
    capture_count = 0
    frame_count = 0
    auto_capture_mode = False
    last_auto_capture = 0
    
    print("\n📸 Press 'c' to capture and auto-label, 'q' to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame!")
            break
        
        frame_count += 1
        
        # Auto-capture mode
        if auto_capture_mode:
            current_time = time.time()
            if current_time - last_auto_capture >= 10:  # Auto-capture every 10 seconds
                print("\n" + "=" * 50)
                print(f"🤖 AUTO-CAPTURE at {datetime.now().strftime('%H:%M:%S')}")
                capture_count, status, confidence, features = capture_and_auto_label(
                    frame, model, feature_names, capture_count
                )
                print("=" * 50)
                last_auto_capture = current_time
        
        # Manual testing every ~5 seconds
        if frame_count % 150 == 0 and not auto_capture_mode:
            print("\n" + "=" * 50)
            print(f"🔍 AUTO-TEST at {datetime.now().strftime('%H:%M:%S')}")
            
            features, bbox = extract_features_from_frame(frame)
            
            if features is not None:
                prediction, probabilities = predict_with_features(model, features, feature_names)
                
                status = "ALIVE" if prediction == 1 else "DEAD"
                dead_conf = probabilities[0] * 100
                alive_conf = probabilities[1] * 100
                max_conf = max(dead_conf, alive_conf)
                
                print(f"📊 Prediction: {status}")
                print(f"   DEAD confidence: {dead_conf:.2f}%")
                print(f"   ALIVE confidence: {alive_conf:.2f}%")
                print(f"   Max confidence: {max_conf:.2f}%")
                
                print("\n📈 Feature values:")
                for name, value in zip(feature_names, features):
                    print(f"   {name}: {value:.4f}")
            else:
                print("❌ No clam detected in frame!")
            
            print("=" * 50)
        
        # Manual input every ~3 seconds
        if frame_count % 90 == 0:
            print("\n🔄 Press 't'=test, 'c'=capture, 's'=save, 'a'=auto, 'q'=quit")
            user_input = input("> ").lower().strip()
            
            if user_input == 't':
                print("\n🔍 MANUAL TEST:")
                features, bbox = extract_features_from_frame(frame)
                
                if features is not None:
                    prediction, probabilities = predict_with_features(model, features, feature_names)
                    
                    status = "ALIVE" if prediction == 1 else "DEAD"
                    dead_conf = probabilities[0] * 100
                    alive_conf = probabilities[1] * 100
                    
                    print(f"   Prediction: {status}")
                    print(f"   DEAD confidence: {dead_conf:.2f}%")
                    print(f"   ALIVE confidence: {alive_conf:.2f}%")
                else:
                    print("❌ No clam detected")
                    
            elif user_input == 'c':
                print("\n📸 CAPTURE AND AUTO-LABEL:")
                capture_count, status, confidence, features = capture_and_auto_label(
                    frame, model, feature_names, capture_count
                )
                
            elif user_input == 's':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_captures/test_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved frame: {filename}")
                capture_count += 1
                
            elif user_input == 'a':
                auto_capture_mode = not auto_capture_mode
                if auto_capture_mode:
                    print("🤖 Auto-capture mode ENABLED (every 10 seconds)")
                    last_auto_capture = time.time()
                else:
                    print("🤖 Auto-capture mode DISABLED")
                    
            elif user_input == 'q':
                print("🛑 Quitting...")
                break
        
        time.sleep(0.1)
    
    cap.release()
    print(f"\n✅ Test complete! Saved {capture_count} captures.")
    print("   Check 'test_captures/' folder for saved images.")
    print("   - 'alive/' folder: Auto-labeled alive clams")
    print("   - 'dead/' folder: Auto-labeled dead clams")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("🐚 Clam Model Test with Auto-Label")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('clam_model_no_overfit.pkl'):
        print("❌ ERROR: 'clam_model_no_overfit.pkl' not found!")
        print("Please run algo.py first to create the model.")
        exit()
    
    # Check if feature names exist
    if not os.path.exists('feature_names.json'):
        print("❌ ERROR: 'feature_names.json' not found!")
        print("Please run algo.py first to create the feature names.")
        exit()
    
    # Use camera 2 (Iriun Webcam)
    test_model_with_webcam(camera_id=2)