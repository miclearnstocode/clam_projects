import cv2
import numpy as np
import os
import csv

base_dir = os.getcwd()
alive_path = os.path.join(base_dir, "train", "alive")
dead_path = os.path.join(base_dir, "train", "dead")

dataset = []

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w, _ = img.shape
    cropped = img[int(h * 0.4):, :]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # --- STRICT PINK NOSE DETECTION (Only Alive has this) ---
    # Pink/Red nose: High Hue (140-180), Medium-High Saturation, Medium-High Value
    lower_pink = np.array([130, 80, 80])  # Stricter Saturation to avoid white shell
    upper_pink = np.array([180, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    # --- WHITE NOSE DETECTION (Dead) ---
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find the body of the clam (the shell)
    body_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
    kernel = np.ones((15, 15), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < 15000:
        return None

    # --- Extract features ---
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Get color stats
    mask_bgr = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask_bgr, [main_contour], -1, 255, -1)
    mean_color = cv2.mean(cropped, mask=mask_bgr)[:3]
    mean_blue, mean_green, mean_red = mean_color

    std_dev = np.std(cropped[mask_bgr == 255], axis=0)
    std_blue, std_green, std_red = std_dev

    # --- CRITICAL: PINK DOMINANCE ---
    # Count pink pixels ONLY inside the clam contour
    pink_in_contour = cv2.countNonZero(cv2.bitwise_and(mask_pink, mask_bgr))
    pink_ratio = pink_in_contour / area if area > 0 else 0

    # Count white pixels ONLY inside the clam contour
    white_in_contour = cv2.countNonZero(cv2.bitwise_and(mask_white, mask_bgr))
    white_ratio = white_in_contour / area if area > 0 else 0

    # --- ENGINEERED FEATURES FOR PINK DETECTION ---
    redness = mean_red - mean_green  # High Red vs Green = Pink
    pink_intensity = mean_red / (mean_blue + 1) # High Red vs Blue = Pink

    # THE MOST IMPORTANT FEATURE: Pink Dominance
    # If there's a pink nose, this value will be HIGH. If no pink, it will be LOW.
    pink_dominance = (pink_ratio * 100) - (white_ratio * 10)  # Weighted heavily for pink

    features = [
        aspect_ratio, circularity, solidity,
        mean_blue, mean_green, mean_red,
        std_blue, std_green, std_red,
        pink_ratio,          # <-- Key feature for ALIVE
        white_ratio,         # <-- Key feature for DEAD
        redness,             # <-- High value = Alive
        pink_intensity,      # <-- High value = Alive
        pink_dominance       # <-- THE ULTIMATE ALIVE INDICATOR
    ]
    return features

def process_folder(folder_path, label, label_name):
    count = 0
    if not os.path.exists(folder_path):
        print(f"❌ WARNING: Folder does not exist: {folder_path}")
        return count

    print(f"Processing {label_name} images from: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            features = extract_features(full_path)
            
            if features is not None:
                features.append(label)
                dataset.append(features)
                count += 1
                print(f"✅ Processed: {filename} ({label_name} #{count})")
            else:
                print(f"⚠️ No clam found in {filename}")
    return count

print("Starting feature extraction...\n")
alive_count = process_folder(alive_path, 1, "Alive")
dead_count = process_folder(dead_path, 0, "Dead")

csv_filename = "clam_dataset_nose.csv"
header = [
    "Aspect_Ratio", "Circularity", "Solidity",
    "Mean_Blue", "Mean_Green", "Mean_Red",
    "Std_Blue", "Std_Green", "Std_Red",
    "Pink_Ratio", "White_Ratio", "Redness", "Pink_Intensity", "Pink_Dominance",
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