import os
from pathlib import Path


folder = Path(r"C:/Users/Cogito/Downloads/alive_images")  # Replace with your folder path
images = sorted(folder.glob("*.*"))  # Get all files, sorted

start_num = 154
end_num = 203

for i, img in enumerate(images):
    current_num = start_num + i
    if current_num > end_num:
        break
    new_name = f"alive_{current_num:03d}{img.suffix}"  # alive_106.jpg, alive_107.jpg, etc.
    img.rename(folder / new_name)
    print(f"Renamed: {img.name} → {new_name}")