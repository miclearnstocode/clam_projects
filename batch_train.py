import cv2
import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =================
ALIVE_FOLDER = "train/alive"   # Folder with alive clams
DEAD_FOLDER = "train/dead"     # Folder with dead clams
OUTPUT_MODEL = "clam_model_engineered.pkl"
# ==========================================

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

def extract_features_from_crop(crop):
    """Extract features from a cropped image (grid cell)."""
    if crop.size == 0:
        return None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Use the full crop as a mask (grid cell is our candidate)
    mask = np.ones(crop.shape[:2], np.uint8) * 255
    
    mean_color = cv2.mean(crop, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(crop[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    h, w = crop.shape[:2]
    aspect_ratio = float(w) / h
    circularity = 0.0
    solidity = 1.0
    
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
    return [features[name] for name in feature_names]

def split_image_into_grid(image, grid_size=(4, 4)):
    """Split image into a grid of crops (candidate boxes)."""
    h, w, _ = image.shape
    grid_h, grid_w = grid_size
    crops = []
    
    cell_h = h // grid_h
    cell_w = w // grid_w
    
    for i in range(grid_h):
        for j in range(grid_w):
            y_start = i * cell_h
            y_end = min((i + 1) * cell_h, h)
            x_start = j * cell_w
            x_end = min((j + 1) * cell_w, w)
            
            # Skip cells that are too small
            if (y_end - y_start) < 50 or (x_end - x_start) < 50:
                continue
                
            crop = image[y_start:y_end, x_start:x_end]
            crops.append((crop, (x_start, y_start, cell_w, cell_h)))
    
    return crops

def process_folder(folder_path, label):
    """Process all images in a folder and return features + labels."""
    data = []
    print(f"Processing folder: {folder_path} (Label: {label})")
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Split into grid cells (boxes)
        crops = split_image_into_grid(img, grid_size=(4, 4))
        
        for crop, bbox in crops:
            features = extract_features_from_crop(crop)
            if features:
                data.append(features + [label])
                print(f"  ✅ Added sample from {filename}")
    
    return data

# ================= MAIN TRAINING =================
print("📸 Starting Batch Training...")

all_data = []

# Process ALIVE images
if os.path.exists(ALIVE_FOLDER):
    all_data.extend(process_folder(ALIVE_FOLDER, 1))
else:
    print(f"⚠️ Folder not found: {ALIVE_FOLDER}")

# Process DEAD images
if os.path.exists(DEAD_FOLDER):
    all_data.extend(process_folder(DEAD_FOLDER, 0))
else:
    print(f"⚠️ Folder not found: {DEAD_FOLDER}")

if len(all_data) < 10:
    print("❌ Not enough samples! Make sure your folders have images.")
    exit()

# Convert to DataFrame
df = pd.DataFrame(all_data, columns=feature_names + ['Label'])
print(f"\n📊 Total samples: {len(df)}")
print(f"   Alive (1): {len(df[df['Label']==1])}")
print(f"   Dead (0): {len(df[df['Label']==0])}")

# Train the model
X = df.drop('Label', axis=1)
y = df['Label']

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X, y)

# Save the model
joblib.dump(model, OUTPUT_MODEL)
print(f"\n✅ Model trained and saved as '{OUTPUT_MODEL}'")
print("Now restart app.py to use the new model!")