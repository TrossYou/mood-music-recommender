# src/analyze_clip.py
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 감정 키워드
MOODS = [ "cozy", "delicious", "excited", "gloomy", "happy", "lovely", "ordinary", "peaceful", "sensitive"]

# 이미지 → 텍스트 유사도 분석
def analyze_clip_similarity_single(image_dir, output_csv, model, processor):
    """
    Analyze CLIP similarity for each image in the directory.
    If with_finetuned is True, run both baseline and finetuned CLIP analysis and save results to seperate CSV files.
    Otherwise, run baselin CLIP only and save to output_csv.
    """
    
    if not any(fname.lower().endswith((".jpg", ".jpeg", ".png")) for fname in os.listdir(image_dir)):
        print(f"[SKIP] No images found in {image_dir}. Skipping analysis.")
        return
    rows = []
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(text=MOODS, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

        # 가장 높은 감정과 점수 계산
        best_idx = logits_per_image.argmax()
        best_mood = MOODS[best_idx]
        best_score = logits_per_image[best_idx]

        row = {"image": filename}
        for i, mood in enumerate(MOODS):
            row[mood] = logits_per_image[i]
        row["best_mood"] = best_mood
        row["best_score"] = best_score

        rows.append(row)

    df = pd.DataFrame(rows)

    # 감정 라벨 컬럼
    df["label"] = df["image"].apply(lambda x: x.split("_")[0])

    # 매칭 여부 컬럼
    df["match"] = df["label"] == df["best_mood"]

    match_count = df["match"].sum()
    
    df.to_csv(output_csv, index=False)

    with open(output_csv, 'a') as f:
        f.write(f"\nmatch_rate,{match_count/len(df):.4f}\n")
    print(f"[✔] Saved to {output_csv}")
        

def analyze_clip_similarity_grouped(image_dir, label_to_images, output_csv):
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rows = []
    for label, image_list in label_to_images.items():
        mood_scores = []

        for image_name in image_list:
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=MOODS, images=image, return_tensors="pt", padding=True).to(device)
            outputs = clip(**inputs)
            logits = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            mood_scores.append(logits)

        # 평균 감정 벡터 계산
        avg_scores = sum(mood_scores) / len(mood_scores)
        best_idx = avg_scores.argmax()
        best_mood = MOODS[best_idx]
        best_score = avg_scores[best_idx]

        row = {"label": label, "best_mood": best_mood, "best_score": best_score}
        for i, mood in enumerate(MOODS):
            row[mood] = avg_scores[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["match"] = df["label"] == df["best_mood"]
    match_rate = df["match"].sum() / len(df)

    df.to_csv(output_csv, index=False)
    with open(output_csv, 'a') as f:
        f.write(f"\nmatch_rate,{match_rate:.4f}\n")

    print(f"[✔] Saved to {output_csv}")

def analyze_clip_score_only(image_path, model=None, processor=None):
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=MOODS, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

    return {mood:logits[i] for i, mood in enumerate(MOODS)}

def analyze_clip_from_file(image_path, model=None, processor=None):
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=MOODS, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

    best_idx = logits.argmax()
    return {
        "best_mood": MOODS[best_idx],
        "best_score": logits[best_idx]
    }