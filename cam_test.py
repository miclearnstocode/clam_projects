import cv2

# Test camera IDs 0, 1, 2, 3, 4...
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Save a test image so you can see the difference
            cv2.imwrite(f"test_camera_{i}.jpg", frame)
            print(f"✅ Camera {i} is working! Saved image to test_camera_{i}.jpg")
        cap.release()
    else:
        print(f"❌ Camera {i} not found")