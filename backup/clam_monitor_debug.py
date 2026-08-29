import cv2
import numpy as np
import joblib
import time
from datetime import datetime
import os

# Load your trained model
try:
    model = joblib.load('clam_model.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: 'clam_model.pkl' not found!")
    print("Please run algo.py first to create the model.")
    exit()

# Feature extraction function with debug output
def extract_features_from_frame(frame, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold (better for varying lighting)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug: print("No contours found")
        return None, None
    
    # Filter contours by size and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000 or area > 100000:  # Adjusted range
            continue
        
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Clams typically have aspect ratio between 0.3 and 3.0
        if 0.3 < aspect_ratio < 3.0:
            valid_contours.append(contour)
            if debug: print(f"Valid contour found: area={area}, aspect_ratio={aspect_ratio:.2f}")
    
    if not valid_contours:
        if debug: print("No valid contours found")
        return None, None
    
    # Get the largest valid contour
    main_contour = max(valid_contours, key=cv2.contourArea)
    
    # Calculate features
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    mean_blue, mean_green, mean_red = mean_color
    
    std_dev = np.std(frame[mask == 255], axis=0)
    std_blue, std_green, std_red = std_dev
    
    features = [aspect_ratio, circularity, solidity, 
                mean_blue, mean_green, mean_red, 
                std_blue, std_green, std_red]
    
    if debug:
        print(f"Features extracted: {features}")
    
    return features, (x, y, w, h)

# Save debug images
def save_debug_image(frame, filename):
    os.makedirs("debug_images", exist_ok=True)
    filepath = os.path.join("debug_images", filename)
    cv2.imwrite(filepath, frame)
    print(f"📸 Saved debug image: {filepath}")

# Main monitoring function
def monitor_clam(camera_id=2, check_interval=30):
    """
    Monitor clam status continuously
    camera_id: 2 for Iriun Webcam
    check_interval: seconds between checks
    """
    
    # Open camera with DShow backend (Windows)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"✅ Iriun Webcam (Camera {camera_id}) opened successfully!")
    print(f"Starting clam monitoring...")
    print(f"Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    last_check_time = 0
    dead_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            # Check at intervals
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                # Save preview frame every time
                save_debug_image(frame, f"preview_{datetime.now().strftime('%H%M%S')}.jpg")
                
                # Extract features with debug
                features, bbox = extract_features_from_frame(frame, debug=True)
                
                if features is not None:
                    # Predict with feature names to avoid warning
                    feature_names = ['aspect_ratio', 'circularity', 'solidity', 
                                   'mean_blue', 'mean_green', 'mean_red', 
                                   'std_blue', 'std_green', 'std_red']
                    
                    # Create DataFrame-like structure
                    features_dict = {name: [value] for name, value in zip(feature_names, features)}
                    
                    # Predict using the features
                    prediction = model.predict([features])[0]
                    probability = model.predict_proba([features])[0]
                    
                    clam_status = "ALIVE" if prediction == 1 else "DEAD"
                    confidence = max(probability) * 100
                    
                    # Print status
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Status: {clam_status} (Confidence: {confidence:.1f}%)")
                    
                    # Save feature details
                    print(f"  Aspect Ratio: {features[0]:.3f}")
                    print(f"  Circularity: {features[1]:.3f}")
                    print(f"  Solidity: {features[2]:.3f}")
                    print(f"  Mean RGB: ({features[3]:.0f}, {features[4]:.0f}, {features[5]:.0f})")
                    
                    # Alert if dead
                    if clam_status == "DEAD" and confidence > 70:
                        dead_counter += 1
                        print(f"⚠️ ALERT: Clam detected as DEAD! (Detection #{dead_counter})")
                        save_debug_image(frame, f"dead_clam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                else:
                    clam_status = "NO CLAM DETECTED"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Status: {clam_status}")
                
                last_check_time = current_time
            
            # Small delay to prevent CPU overuse
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    
    cap.release()
    print(f"\nMonitoring stopped. Total dead detections: {dead_counter}")

# Run the monitor
if __name__ == "__main__":
    print("🔍 Starting Clam Monitoring with Iriun Webcam (Camera 2)...")
    print("Make sure Iriun Webcam is running on your iPhone and connected via USB")
    print()
    
    # Check if model exists
    if not os.path.exists('clam_model.pkl'):
        print("❌ ERROR: 'clam_model.pkl' not found!")
        print("Please run algo.py first to create the model.")
        exit()
    
    # Start monitoring with debug
    monitor_clam(camera_id=2, check_interval=30)