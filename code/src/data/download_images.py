import os
import subprocess

def download_images_from_drive(folder_id, output_dir):
    if os.path.exists(output_dir) and any(
        fname.lower().endswith((".jpg", ".jpeg", ".png")) for fname in os.listdir(output_dir)
    ):
        print(f"[SKIP 0/4] {output_dir} already contains files. Skipping YOLO cropping for person.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[0/4] Downloading images from Google Drive folder to {output_dir}...")
    try:
        result = subprocess.run(
            ['gdown', '--folder', folder_id, '--output', output_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        if not os.listdir(output_dir):  # 폴더가 비어 있으면 실패
            raise RuntimeError(f"[ERROR] No files downloaded to {output_dir}. Check folder ID or permissions.")
        print("[✔] Download completed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download images from Google Drive: {e.stderr}")
        exit(1)