import cv2

def test_cameras_no_gui():
    working_cameras = []
    for i in range(20):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera {i} is working! Resolution: {width}x{height}")
                working_cameras.append(i)
        cap.release()
    
    print(f"\nFound {len(working_cameras)} working cameras: {working_cameras}")
    return working_cameras

# Find all working cameras
cameras = test_cameras_no_gui()

# Save the first frame from each camera for verification
def save_camera_test():
    for i in cameras:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Save as image file instead of showing
                cv2.imwrite(f"camera_{i}_test.jpg", frame)
                print(f"Saved camera_{i}_test.jpg - check this image to see if it's Iriun")
        cap.release()

if cameras:
    save_camera_test()
    print("\n✅ Test complete! Check the saved images to find Iriun Webcam.")