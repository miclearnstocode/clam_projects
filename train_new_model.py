import cv2
import numpy as np
import pandas as pd
import os
import joblib
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =================== FOLDER PATHS ===================
ALIVE_FOLDER = "train/alive"
DEAD_FOLDER = "train/dead"
NO_CLAM_FOLDER = "train/no_clam"
# ====================================================

# Load existing feature names (or create defaults)
try:
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
except:
    feature_names = [
        'Aspect_Ratio', 'Circularity', 'Solidity',
        'Mean_Blue', 'Mean_Green', 'Mean_Red',
        'Std_Blue', 'Std_Green', 'Std_Red',
        'Blue_Green_Ratio', 'Green_Red_Ratio', 'Blue_Minus_Green',
        'Aspect_Circularity', 'Shape_Score',
        'Total_Color_Variation', 'Color_Variation_Product',
        'Mean_Green_Normalized'
    ]

def safe_std_dev(values):
    """Safely get std dev, even if array is empty."""
    if len(values) == 0:
        return [0.0, 0.0, 0.0]
    values = np.array(values)
    if values.ndim == 1:
        return [float(np.std(values)), 0.0, 0.0]
    return [float(np.std(values[:, i])) if len(values) > 0 else 0.0 for i in range(3)]

def extract_features_from_frame(frame):
    """Extracts features using the Siphon + Edge detection logic."""
    h, w, _ = frame.shape

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
    
    # 5. Fuse edges with saturated image
    edges_bgr = cv2.cvtColor(edges_thresh, cv2.COLOR_GRAY2BGR)
    fused = cv2.addWeighted(frame_sat, 0.8, edges_bgr, 0.2, 0)
    
    # 6. Find Pink/Violet siphon mask
    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(hsv, (140, 90, 90), (180, 255, 255))
    
    # 7. Morphology to connect the siphon
    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    
    # 8. Find contours (if any)
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no siphon, use the whole image for feature extraction (background)
    if not contours:
        mean_color = cv2.mean(frame)[:3]  # Mean of whole frame
        std_dev = [float(np.std(frame[:,:,0])), float(np.std(frame[:,:,1])), float(np.std(frame[:,:,2]))]
        aspect_ratio = 1.0
        circularity = 0.0
        solidity = 0.0
    else:
        # Pick the largest pink contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        
        # Get mean color and std dev of the siphon region
        mask_bgr = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask_bgr, [contour], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask_bgr)[:3]
        
        # Safe std dev calculation
        pixels = frame[mask_bgr == 255]
        std_dev = safe_std_dev(pixels)
    
    # Engineered features
    mean_blue, mean_green, mean_red = mean_color
    std_blue, std_green, std_red = std_dev
    
    blue_green_ratio = mean_blue / (mean_green + 1)
    green_red_ratio = mean_green / (mean_red + 1)
    blue_minus_green = mean_blue - mean_green
    aspect_circularity = aspect_ratio * circularity
    shape_score = aspect_ratio / (circularity + 0.01)
    total_color_variation = std_blue + std_green + std_red
    color_variation_product = std_blue * std_green * std_red
    mean_green_normalized = mean_green / (mean_blue + mean_green + mean_red)
    
    return [
        aspect_ratio, circularity, solidity,
        mean_blue, mean_green, mean_red,
        std_blue, std_green, std_red,
        blue_green_ratio, green_red_ratio, blue_minus_green,
        aspect_circularity, shape_score,
        total_color_variation, color_variation_product,
        mean_green_normalized
    ]

# ==========================================================
# 1. PROCESS FOLDERS
# ==========================================================
all_data = []
count = 0

def process_folder(folder_path, label):
    global count
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return 0
    
    print(f"Processing: {folder_path} (Label: {label})")
    folder_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feats = extract_features_from_frame(img)
            all_data.append(feats + [label])
            folder_count += 1
            count += 1
            if folder_count % 10 == 0:
                print(f"   ✅ Processed {folder_count} images from {folder_path}")
    
    return folder_count

alive_count = process_folder(ALIVE_FOLDER, 1)  # 1 = Alive
dead_count = process_folder(DEAD_FOLDER, 0)    # 0 = Dead
no_clam_count = process_folder(NO_CLAM_FOLDER, 2) # 2 = No Clam

print(f"\n✅ Collected: Alive={alive_count}, Dead={dead_count}, NoClam={no_clam_count}")
print(f"Total samples: {count}")

# ==========================================================
# 2. TRAIN MODEL
# ==========================================================
if len(all_data) < 20:
    print("❌ Not enough samples! Need at least 20 total images.")
    exit()

df = pd.DataFrame(all_data, columns=feature_names + ['Label'])
X = df.drop('Label', axis=1)
y = df['Label']

# Train a 3-class model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
    class_weight={0: 1.0, 1: 1.5, 2: 1.0}  # Give ALIVE slightly more weight
)
model.fit(X, y)

# ==========================================================
# 3. SAVE MODEL & FEATURE NAMES
# ==========================================================
joblib.dump(model, 'clam_model_engineered.pkl')
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print(f"\n✅ Model trained and saved to 'clam_model_engineered.pkl'")
print(f"✅ Feature names saved to 'feature_names.json'")
print("Restart app.py or test_lab.py to use the new model!")