from ultralytics import YOLO

if __name__ == "__main__":
    # 'n' = nano (smallest, fastest on CPU). Good for ~300 images.
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=80,               # 80 is plenty for ~250 train images
        imgsz=480,               # lower than 640 = 2.5x faster on CPU
        batch=8,                 # small batch works well on CPU
        name="pholas_monitor",
        patience=15,             # stop early if no improvement
        workers=2,               # Windows-safe
        device="cpu",            # Intel Iris Xe → use CPU
        cache=True,              # cache images in RAM (small dataset = fits!)
        plots=True,
        augment=True,            # helps a lot with only 310 images
        # Augmentation settings (mild, to avoid over-distorting clams)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,            # small rotations
        translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.0,  # clams aren't upside-down
        mosaic=1.0,
        mixup=0.0,               # off — too aggressive for small data
    )

    metrics = model.val()
    print("\n=== FINAL METRICS ===")
    print("mAP50:    ", metrics.box.map50)
    print("mAP50-95: ", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:   ", metrics.box.mr)