import os
import zipfile

# Path to your main dataset folder (change if needed)
project_root = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.environ.get("BRATS_ARCHIVE_DIR", os.path.join(project_root, "archive"))

# Recursively go through all folders
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".nii.zip") or file.endswith(".gz") or file.endswith(".zip"):
            file_path = os.path.join(root, file)
            print(f"Extracting: {file_path}")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
            except zipfile.BadZipFile:
                print(f"⚠ Not a zip file: {file_path}")

print("✅ All files extracted!")
