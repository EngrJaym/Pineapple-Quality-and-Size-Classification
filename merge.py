import os
import shutil
import glob
from pathlib import Path

def consolidate_images_unique(source_directory, destination_folder_name="All_Images_Consolidated"):
    """
    Recursively finds all images in the source_directory and its subfolders,
    creates a new destination folder, and copies all images into it,
    renaming files if duplicates exist.

    Args:
        source_directory (str): The path to the directory to search for images.
        destination_folder_name (str): The name of the new folder to create.
    """
    source_path = Path(source_directory)
    destination_path = source_path / destination_folder_name

    # Create the destination directory if it doesn't exist
    try:
        os.makedirs(destination_path, exist_ok=True)
        print(f"Created destination folder: {destination_path}")
    except OSError as e:
        print(f"Error creating destination folder {destination_path}: {e}")
        return

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff')
    copied_count = 0

    print(f"Searching for images in: {source_directory} and subfolders...")

    for ext in image_extensions:
        for file_path in glob.iglob(str(source_path / '**' / ext), recursive=True):
            original_file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(original_file_name)
            destination_file_path = destination_path / original_file_name
            counter = 1

            # Check if file exists and generate a unique name if it does
            while os.path.exists(destination_file_path):
                new_file_name = f"{file_base}_{counter}{file_ext}"
                destination_file_path = destination_path / new_file_name
                counter += 1

            try:
                # Copy the file to the destination folder with the determined unique name
                shutil.copy2(file_path, destination_file_path)
                copied_count += 1
                if counter > 1:
                    print(f"Renamed and copied: {original_file_name} -> {new_file_name}")
                # else:
                #     print(f"Copied: {original_file_name}")
            except Exception as e:
                print(f"Error copying file {file_path}: {e}")

    print(f"\nFinished copying. Total images copied: {copied_count}")

# --- How to use the script ---

# 1. Specify the source directory where your images are located
#    Example: r"C:\Users\YourUser\Pictures\VacationPhotos"
source_directory_path = r"C:\Users\NDS.BPAA\Downloads\archive (1)\validation" # !! REPLACE THIS PATH !!

# 2. Call the function
if source_directory_path == r"/path/to/your/main/image/folder":
    print("Please update the 'source_directory_path' variable in the script to run it.")
else:
    consolidate_images_unique(source_directory_path)

