import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

def crop_mtcnn(image_path, mtcnn_person_dir):
    mtcnn = MTCNN(keep_all=True)
    os.makedirs(mtcnn_person_dir, exist_ok=True)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Could not open image: {image_path} - {e}")
        return
    
    faces = mtcnn(image)
    if faces is None:
        print(f"[INFO] No faces found in image: {image_path}")
        return
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for j, face in enumerate(faces):
        if face is None:
            continue
        face_image = transforms.ToPILImage()(face)
        name = f"{base_name}_mtcnn{j}.jpg"
        face_image.save(os.path.join(mtcnn_person_dir, name))
    
def run_mtcnn_person(image_dir, mtcnn_person_dir):
    if os.path.exists(mtcnn_person_dir) and any(
        fname.lower().endswith((".jpg", ".jpeg", ".png")) for fname in os.listdir(mtcnn_person_dir)
    ):
        print(f"[SKIP 2/4] {mtcnn_person_dir} already contains files. Skipping MTCNN cropping for person.")
        return 
    
    print("[2/4] Running MTCNN cropping for person...")
    os.makedirs(mtcnn_person_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            crop_mtcnn(image_path, mtcnn_person_dir)
                

                
                
