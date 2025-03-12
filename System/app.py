import os
import shutil
import cv2
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# Load the YOLO model
model = YOLO(r'D:\My Laptop\Me\Programming\Machine Learning\Internships\Egyptian Space Agency\YOLO\yolov11\yolov11\new_x_large_model\weights\best.pt')

# Define directories
both_dir = 'both'
type2_dir = 'Type_2'
type3_dir = 'Type_3'
annotated_dir = 'annotated'
input_dir = 'data'
no_bursts = 'no_bursts'

# Create output directories if they don't exist
for folder in [both_dir, type2_dir, type3_dir, annotated_dir, no_bursts]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to process an image
def process_image(image_path):
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Run the model on the image
    result = model(img)
    
    # Extract detected classes and bounding boxes
    detected_classes = [model.names[int(cls)] for cls in result[0].boxes.cls]
    bboxes = result[0].boxes.xywhn  # Normalized bounding boxes (center_x, center_y, width, height)
    class_ids = result[0].boxes.cls  # Class indices
    
    # Save YOLO annotation
    annotation_path = os.path.join(annotated_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(annotation_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_ids):
            cls_id = int(cls_id)  # Convert class ID to integer
            bbox_str = ' '.join(map(str, bbox.tolist()))  # Convert bbox to string
            f.write(f"{cls_id} {bbox_str}\n")
    print(f"Saved annotation for {image_name} at {annotation_path}")
    
    # Check conditions and move files to appropriate folders
    if 'TYPE 2' in detected_classes and 'TYPE 3' in detected_classes:
        destination_path = os.path.join(both_dir, image_name)
        shutil.move(image_path, destination_path)
        print(f"Moved {image_name} to {both_dir}")
    elif 'TYPE 2' in detected_classes:
        destination_path = os.path.join(type2_dir, image_name)
        shutil.move(image_path, destination_path)
        print(f"Moved {image_name} to {type2_dir}")
    elif 'TYPE 3' in detected_classes:
        destination_path = os.path.join(type3_dir, image_name)
        shutil.move(image_path, destination_path)
        print(f"Moved {image_name} to {type3_dir}")
    else:
        destination_path = os.path.join(no_bursts, image_name)
        shutil.move(image_path, destination_path)
        print(f"Moved {image_name} to {no_bursts}")

# Process existing images in the folder
print("Processing existing images...")
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    process_image(image_path)

# Define the event handler
class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_image(event.src_path)

# Set up the observer
observer = Observer()
event_handler = NewImageHandler()
observer.schedule(event_handler, input_dir, recursive=False)

try:
    print(f"Monitoring folder: {input_dir}")
    observer.start()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping observer...")
    observer.stop()

observer.join()
