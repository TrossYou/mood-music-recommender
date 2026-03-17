import os
import cv2
from ultralytics import YOLO
from PIL import Image

def crop_yolo(image_path, yolo_person_dir, model_path='yolov8n.pt'):
    yolo = YOLO(model_path)
    os.makedirs(yolo_person_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image: {image_path}")
        return
    
    results = yolo.predict(image, conf=0.5)

    for j, result in enumerate(results):
        boxes = result.boxes
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()
            cls = boxes.cls[j].item()
            if int(cls) != 0:
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            name = f"{base_name}_yolo{j}.jpg"
            cropped_image.save(os.path.join(yolo_person_dir, name))

def run_yolo_person(image_dir, yolo_person_dir, model_path='yolov8n.pt'):
    if os.path.exists(yolo_person_dir) and any(
        fname.lower().endswith((".jpg", ".jpeg", ".png")) for fname in os.listdir(yolo_person_dir)
    ):
        print(f"[SKIP 1/4] {yolo_person_dir} already contains files. Skipping YOLO cropping for person.")
        return

    print("[1/4] Running YOLO cropping for person...")
    os.makedirs(yolo_person_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            crop_yolo(image_path, yolo_person_dir, model_path)