import os
import random
import shutil
from pathlib import Path

# Paths
img_dir = Path("/home/hunglt31/examark/dataset/content_ori/train/images")
lbl_dir = Path("/home/hunglt31/examark/dataset/content_ori/train/labels")
output_base = Path("/home/hunglt31/examark/dataset/content")
train_img, val_img, test_img = output_base / "train" / "images", output_base / "valid" / "images", output_base / "test" / "images"
train_lbl, val_lbl, test_lbl = output_base / "train" / "labels", output_base / "valid" / "labels", output_base / "test" / "labels"

# Create dirs
for d in [train_img, val_img, test_img, train_lbl, val_lbl, test_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Gather and shuffle images
image_files = list(img_dir.glob("*.jpg")) 
random.shuffle(image_files)

for d in [train_img, val_img, test_img, train_lbl, val_lbl, test_lbl]:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)

# Split
n = len(image_files)
train_split = int(n * 0.8)
train_set = image_files[:train_split]
val_set = image_files[train_split:]

# Helper to copy files
def copy_files(file_list, img_dst, lbl_dst):
    for img_path in file_list:
        label_path = lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, img_dst / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, lbl_dst / label_path.name)

# Copy to folders
copy_files(train_set, train_img, train_lbl)
copy_files(val_set, val_img, val_lbl)