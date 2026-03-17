import os
import cv2
import numpy as np
from ultralytics import YOLO

def crop_yolo_background(image_path, yolo_background_dir, model_path='yolov8n.pt'):
    yolo = YOLO(model_path)
    os.makedirs(yolo_background_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image: {image_path}")
        return

    results = yolo.predict(image, conf=0.5)
    masked_image = image.copy()  

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for result in results:
        boxes = result.boxes 
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])  
            cls = int(box.cls.item())
            if cls != 0:  
                continue
            
            mask[y1:y2, x1:x2] = 255

    if np.count_nonzero(mask) == 0:
        print(f"[INFO] No person detected in {image_path}. Using original image as background.")
        masked_image = image.copy()
    else:
        masked_image[mask == 255] = (127, 127, 127)  

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    name = f"{base_name}_masked.jpg"
    
    cv2.imwrite(os.path.join(yolo_background_dir, name), masked_image)

def run_yolo_background(image_dir, yolo_background_dir, model_path='yolov8n.pt'):
    if os.path.exists(yolo_background_dir) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(yolo_background_dir)
    ):
        print(f"[SKIP 2/4] {yolo_background_dir} already contains files. Skipping YOLO background masking.")
        return

    print("[2/4] Running YOLO background masking...")
    os.makedirs(yolo_background_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            crop_yolo_background(image_path, yolo_background_dir, model_path)