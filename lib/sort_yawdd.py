import os
import shutil

# Source folder containing YawDD videos
source_dir = "/Users/carney/Downloads/YawDD/Mirror/Mirror/Male_mirror Avi Videos"
# Destination folder for Normal videos
target_dir = "/Users/carney/Downloads/YawDD/Selected_Normal_Dash"

# Debug: print paths and existence
print("Source directory:", source_dir)
print("Source exists:", os.path.exists(source_dir))
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

# Create target folder if it doesn't exist
os.makedirs(target_dir, exist_ok=True)
print("Target directory:", target_dir)

# List all files in source
all_files = os.listdir(source_dir)
print(f"Total files in source: {len(all_files)}")

# Counter for copied files
count = 0

# Loop through files and copy only Normal videos
for fname in all_files:
    print("Checking:", fname)
    if "Normal" in fname and fname.lower().endswith((".mp4", ".avi")):
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)  # copy2 preserves metadata
        count += 1
        print(f"Copied: {fname}")

print(f"Total Normal videos copied: {count}")