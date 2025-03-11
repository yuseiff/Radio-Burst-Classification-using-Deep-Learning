import os
import json
import cv2
import yaml
import shutil
from glob import glob
import random

# Define paths
base_dir = r"D:\My Laptop\Me\Programming\Machine Learning\Internships\Egyptian Space Agency\YOLO\Data"
classes = ["TYPE 2", "TYPE 3"]  # Add other classes if needed
output_yolo_dir = os.path.join(base_dir, "YOLO_Annotations")
train_images_dir = os.path.join(base_dir, "train/images")
val_images_dir = os.path.join(base_dir, "val/images")
train_labels_dir = os.path.join(base_dir, "train/labels")
val_labels_dir = os.path.join(base_dir, "val/labels")

# Create directories if they don't exist
os.makedirs(output_yolo_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Map labels to class IDs
label_to_id = {
    "TYPE 2": 0,
    "TYPE 3": 1
}

# Function to convert polygon to bounding box
def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)
    return min_x, min_y, max_x, max_y

# Function to create YOLO annotations
def create_yolo_annotations(json_file, output_dir, label_to_id):
    with open(json_file, "r") as f:
        data = json.load(f)

    # Verify image file exists
    image_path = os.path.join(os.path.dirname(json_file), data["imagePath"])
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}. Skipping.")
        return False

    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}. Skipping.")
        return False
    height, width, _ = image.shape

    yolo_annotations = []
    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        if label not in label_to_id:
            continue

        # Convert polygon to bounding box
        min_x, min_y, max_x, max_y = polygon_to_bbox(points)
        center_x = (min_x + max_x) / 2 / width
        center_y = (min_y + max_y) / 2 / height
        box_width = (max_x - min_x) / width
        box_height = (max_y - min_y) / height

        class_id = label_to_id[label]
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

    # Save annotations
    txt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")
    with open(txt_file, "w") as txt:
        txt.write("\n".join(yolo_annotations))

    return True


# Process annotations
error_files = []
for cls in classes:
    class_dir = os.path.join(base_dir, cls, cls)
    for file in os.listdir(class_dir):
        if file.endswith(".json"):
            json_path = os.path.join(class_dir, file)
            success = create_yolo_annotations(json_path, output_yolo_dir, label_to_id)
            if not success:
                error_files.append(json_path)
                
# Save the error files list
with open("error_files.txt", "w") as f:
    for error_file in error_files:
        f.write(f"{error_file}\n")
    f.write(f"The number of error files: {len(error_files)}\n")

print(f"Conversion completed. YOLO annotations are saved in {output_yolo_dir}.")
print(f"Files that couldn't be processed are listed in 'error_files.txt'.")

# Define paths
base_dir = r"D:\My Laptop\Me\Programming\Machine Learning\Internships\Egyptian Space Agency\YOLO\Data"
classes = ["TYPE 2", "TYPE 3"]  # Add other classes if needed
output_yolo_dir = os.path.join(base_dir, "YOLO_Annotations")
train_images_dir = os.path.join(base_dir, "train/images")
val_images_dir = os.path.join(base_dir, "val/images")
train_labels_dir = os.path.join(base_dir, "train/labels")
val_labels_dir = os.path.join(base_dir, "val/labels")

# Splitting images and annotations
train_ratio = 0.8
all_images = glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
random.shuffle(all_images)
split_idx = int(train_ratio * len(all_images))

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]


for img in train_images:
    shutil.copy(img, train_images_dir)
    annotation_file = os.path.join(output_yolo_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
    if os.path.exists(annotation_file):
        shutil.copy(annotation_file, train_labels_dir)
        
for img in val_images:
    shutil.copy(img, val_images_dir)
    annotation_file = os.path.join(output_yolo_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
    if os.path.exists(annotation_file):
        shutil.copy(annotation_file, val_labels_dir)

print("Data split completed.")

# Create data.yaml
data_yaml = {
    'names': classes,
    'nc': len(classes),
    'train': train_images_dir,
    'train_labels': train_labels_dir,
    'val': val_images_dir,
    'val_labels': val_labels_dir
}
yaml_path = os.path.join(base_dir, "data.yaml")
with open(yaml_path, 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)

print(f"data.yaml file saved at {yaml_path}.")

