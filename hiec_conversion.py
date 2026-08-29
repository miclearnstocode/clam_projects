from PIL import Image
import pillow_heif
import os
import sys
from pathlib import Path

def batch_convert_heic_to_jpg(input_folder, output_folder=None, quality=95, delete_original=False):
    """
    Convert all HEIC files in a folder to JPG format
    
    Args:
        input_folder: Folder containing HEIC files
        output_folder: Output folder (if None, saves in same folder as HEIC files)
        quality: JPG quality (1-100, default 95)
        delete_original: If True, deletes original HEIC files after conversion
    """
    # Register HEIF opener with Pillow
    pillow_heif.register_heif_opener()
    
    # Convert input folder to Path object
    input_path = Path(input_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist!")
        return
    
    # Create output folder if specified and doesn't exist
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all HEIC files (case insensitive)
    heic_files = []
    for ext in ['*.heic', '*.HEIC', '*.Heic', '*.heif', '*.HEIF']:
        heic_files.extend(input_path.glob(ext))
    
    # Remove duplicates (if any)
    heic_files = list(set(heic_files))
    
    if not heic_files:
        print(f"No HEIC files found in '{input_folder}'")
        return
    
    print(f"Found {len(heic_files)} HEIC file(s) to convert\n")
    
    converted_count = 0
    failed_count = 0
    skipped_count = 0
    
    for heic_file in heic_files:
        try:
            # Create output filename
            jpg_filename = heic_file.stem + '.jpg'
            jpg_path = output_path / jpg_filename
            
            # Check if output file already exists
            if jpg_path.exists():
                print(f"⚠ Skipped (already exists): {heic_file.name}")
                skipped_count += 1
                continue
            
            # Open and convert HEIC file
            print(f"Converting: {heic_file.name}...", end=' ')
            
            with Image.open(heic_file) as img:
                # Convert to RGB mode (required for JPG)
                img = img.convert('RGB')
                # Save as JPG
                img.save(jpg_path, 'JPEG', quality=quality)
            
            print(f"✓ Done -> {jpg_filename}")
            converted_count += 1
            
            # Optionally delete original HEIC file
            if delete_original:
                heic_file.unlink()
                print(f"  Deleted original: {heic_file.name}")
                
        except Exception as e:
            print(f"✗ Failed - {e}")
            failed_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"✓ Successfully converted: {converted_count}")
    print(f"⚠ Skipped (already exists): {skipped_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"Total files processed: {len(heic_files)}")
    
    if output_folder:
        print(f"\nOutput folder: {output_folder}")
    else:
        print(f"\nOutput folder: {input_folder} (same as input)")

# ============ USAGE ============
if __name__ == "__main__":
    # EDIT THESE VALUES:
    INPUT_FOLDER = "./Part"  # Change this to your folder path
    OUTPUT_FOLDER = "./alive2"  # Set to None to save in same folder
    QUALITY = 95                    # JPG quality (1-100)
    DELETE_ORIGINAL = False         # Set to True to delete HEIC files after conversion
    
    # Run the conversion
    batch_convert_heic_to_jpg(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        quality=QUALITY,
        delete_original=DELETE_ORIGINAL
    )