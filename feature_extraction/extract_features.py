import cv2
import numpy as np
import os
import csv

# Get the current working directory (where you run the script)
base_dir = os.getcwd()

# Build correct paths based on the current directory
alive_path = os.path.join(base_dir, "train", "alive")
dead_path = os.path.join(base_dir, "train", "dead")
no_clam_path = os.path.join(base_dir, "train", "no_clam")

dataset = []

def get_clam_contour(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > 5000:  # Minimum size filter
            return main_contour
    return None

def extract_features(contour, image_path):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_color = cv2.mean(img, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(img[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    # Engineered features
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

def process_folder(folder_path, label, label_name):
    count = 0
    if not os.path.exists(folder_path):
        print(f"❌ WARNING: Folder does not exist: {folder_path}")
        return count

    print(f"Processing {label_name} images from: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            contour = get_clam_contour(full_path)
            
            if contour is not None:
                features = extract_features(contour, full_path)
                features.append(label)
                dataset.append(features)
                count += 1
                print(f"✅ Processed: {filename} ({label_name} #{count})")
            else:
                # If it's a no_clam image, we still add it using zeros for features
                if label == 2:
                    features = [0] * 17  # 17 feature columns
                    features.append(label)
                    dataset.append(features)
                    count += 1
                    print(f"✅ Processed: {filename} ({label_name} #Empty/No Contour {count})")
                else:
                    print(f"⚠️ No clam found in {filename}")
    return count

print("Starting feature extraction...\n")
alive_count = process_folder(alive_path, 1, "Alive")
dead_count = process_folder(dead_path, 0, "Dead")
no_clam_count = process_folder(no_clam_path, 2, "No Clam")

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

print(f"\n✅ DONE! Created {csv_filename}")
print(f"   Total samples: {len(dataset)}")
print(f"   Alive: {alive_count}")
print(f"   Dead: {dead_count}")
print(f"   No Clam: {no_clam_count}")