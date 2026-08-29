import cv2
import numpy as np
import os
import csv

alive_path = "/train/alive"  
dead_path = "/train/dead"

# List to hold all data
dataset = []

# Function to find the clam contour
def get_clam_contour(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold (better for varying lighting)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > 5000:  # Minimum size filter
            return main_contour
    return None

# Function to calculate features (with engineered features)
def extract_features(contour, image_path):
    # Basic features
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Color features
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_color = cv2.mean(img, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(img[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    # --- ENGINEERED FEATURES ---
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

print("🔍 MONITORING CLAM STATUS...")
print("📸 Processing clam images for classification...\n")

alive_count = 0
dead_count = 0
total_processed = 0

# Process ALIVE images
print("🟢 SCANNING: Checking if clam appears ALIVE...")
for filename in os.listdir(alive_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        full_path = os.path.join(alive_path, filename)
        contour = get_clam_contour(full_path)
        
        if contour is not None:
            features = extract_features(contour, full_path)
            features.append(1)  # Label = 1 for Alive
            dataset.append(features)
            alive_count += 1
            total_processed += 1
            print(f"   ✅ MONITORING: Clam detected - STATUS: ALIVE (Capture #{total_processed})")
        else:
            print(f"   ❌ WARNING: Clam NOT detected in frame {filename}")

# Process DEAD images
print("\n🔴 SCANNING: Checking if clam appears DEAD...")
for filename in os.listdir(dead_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        full_path = os.path.join(dead_path, filename)
        contour = get_clam_contour(full_path)
        
        if contour is not None:
            features = extract_features(contour, full_path)
            features.append(0)  # Label = 0 for Dead
            dataset.append(features)
            dead_count += 1
            total_processed += 1
            print(f"   ❌ MONITORING: Clam detected - STATUS: DEAD (Capture #{total_processed})")
        else:
            print(f"   ❌ WARNING: Clam NOT detected in frame {filename}")

csv_filename = "clam_dataset_202.csv"
header = [
    "Aspect_Ratio", "Circularity", "Solidity",
    "Mean_Blue", "Mean_Green", "Mean_Red",
    "Std_Blue", "Std_Green", "Std_Red",
    "Blue_Green_Ratio", "Green_Red_Ratio", "Blue_Minus_Green",
    "Aspect_Circularity", "Shape_Score",
    "Total_Color_Variation", "Color_Variation_Product",
    "Mean_Green_Normalized",
    "Label"
]

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(dataset)

print("\n" + "="*60)
print("📊 MONITORING SUMMARY")
print("="*60)
print(f"   ✅ Total captures processed: {total_processed}")
print(f"   🟢 Alive status detected: {alive_count} times")
print(f"   🔴 Dead status detected: {dead_count} times")
print(f"\n   📁 Dataset saved as: {csv_filename}")
print("   🔍 Monitoring complete!")
print("="*60)