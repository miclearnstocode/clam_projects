import cv2
import numpy as np
import pandas as pd
import joblib
import os, json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# CAMERA ID
CAMERA_ID = 4

# Load Feature Names (Must exist!)
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# --- EXACT COPY of detect_multiple_clams from app.py ---
def detect_multiple_clams(frame):
    h, w, _ = frame.shape
    cropped = frame[int(h * 0.50):, :]
    cropped_h, cropped_w, _ = cropped.shape
    
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower_clam = np.array([0, 15, 40])
    upper_clam = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_clam, upper_clam)
    
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_clams = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        if area < 15000 or area > 150000:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.60:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue
        
        # REJECT PIPES (High Solidity!)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity > 0.95:
            continue
        
        y += int(h * 0.50)
        
        mask_bgr = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask_bgr, [contour], -1, 255, -1)
        mean_color = cv2.mean(cropped, mask=mask_bgr)[:3]
        mean_blue, mean_green, mean_red = mean_color
        
        std_dev = np.std(cropped[mask_bgr == 255], axis=0)
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
        detected_clams.append(([features[name] for name in feature_names], contour, (x, y, w, h)))
    
    detected_clams.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
    return detected_clams[:3]

# --- COLLECT DATA AND RETRAIN ---
def collect_and_train():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📸 Collecting ALIVE data... Point camera at ALIVE clams.")
    print("Press 'a' to capture ALIVE, 'd' to capture DEAD, 'q' to stop.")
    
    data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detected = detect_multiple_clams(frame)
        
        for features, contour, bbox in detected:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            if detected:
                for features, _, _ in detected:
                    data.append(features + [1])  # Label 1 = ALIVE
                print(f"✅ Captured {len(detected)} ALIVE clams")
        elif key == ord('d'):
            if detected:
                for features, _, _ in detected:
                    data.append(features + [0])  # Label 0 = DEAD
                print(f"⚠️ Captured {len(detected)} DEAD clams")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(data) < 10:
        print("❌ Not enough data! Try again.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names + ['Label'])
    print(f"✅ Collected {len(df)} samples!")
    
    # Train new model
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    # Save
    joblib.dump(model, 'clam_model_engineered.pkl')
    print("✅ Model retrained and saved!")
    print("Now restart app.py and it will correctly detect clams!")

# Run it
collect_and_train()