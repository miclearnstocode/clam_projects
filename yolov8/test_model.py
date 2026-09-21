from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/pholas_monitor-2/weights/best.pt")

    # Run on the 30 test images
    results = model.predict(
        source="test/images",
        conf=0.4,
        save=True,
        project="runs/detect",
        name="pholas_test",
    )
    print(f"\nDone. {len(results)} images processed.")
    print("Open: runs/detect/pholas_test/")