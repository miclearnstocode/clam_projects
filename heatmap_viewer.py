import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def apply_pink_enhancement(frame):
    # 1. Contrast & Saturation Boost (Do this FIRST)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 100, 0, 255)  # Saturation +100
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=0)

    # 2. Roberts Cross Edge Detection (FIRST)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roberts_cross_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_cross_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    edge_x = cv2.filter2D(gray, -1, roberts_cross_x)
    edge_y = cv2.filter2D(gray, -1, roberts_cross_y)
    edges = np.sqrt(edge_x**2 + edge_y**2)
    edges = np.uint8(np.clip(edges, 0, 255))
    # Roberts Cross thresholding at 100
    _, edges_thresh = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # 3. Edge Enhancement 5x5 (Smooth the edges)
    kernel = np.ones((5, 5), np.float32) / 25
    edges_smooth = cv2.filter2D(edges_thresh, -1, kernel)

    # 4. SOLARIZE (Only applies to bright areas to create a "glow")
    frame_solarized = cv2.bitwise_not(frame)
    mask = frame > 200  # Only invert VERY bright pixels
    frame[mask] = frame_solarized[mask]

    # 5. GAMMA 3.94 (But now we APPLY it to the EDGES, not the original!)
    gamma = 3.94
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    edges_boosted = cv2.LUT(edges_smooth, table)

    # 6. FUSE the Edges with the Saturated image
    # The pink tip has a "hot edge" that gives it a glowing outline
    edges_boosted_bgr = cv2.cvtColor(edges_boosted, cv2.COLOR_GRAY2BGR)
    fused_frame = cv2.addWeighted(frame, 0.8, edges_boosted_bgr, 0.2, 0)

    # 7. TUNE THE MASK (Target ONLY the high saturation Pink/Violet)
    hsv = cv2.cvtColor(fused_frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(hsv, (140, 90, 90), (180, 255, 255))

    # 8. Morphology to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)

    return fused_frame, pink_mask

def show_siphon_heatmap(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found!")
        return

    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    enhanced_img, pink_mask = apply_pink_enhancement(img)
    heatmap = cv2.applyColorMap(pink_mask, cv2.COLORMAP_JET)

    # Combine: Left (Original), Center (Enhance), Right (Heatmap)
    combined = np.hstack((img, enhanced_img, heatmap))

    output_filename = "siphon_heatmap_result.jpg"
    cv2.imwrite(output_filename, combined)
    print(f"✅ Saved to: {os.path.abspath(output_filename)}")
    print("Open this file to see the enhanced image and heatmap!")

def pick_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Clam Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        show_siphon_heatmap(file_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    pick_image()