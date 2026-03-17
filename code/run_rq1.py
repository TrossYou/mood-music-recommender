import os
import pandas as pd
from src.models.load_models import load_baseline_model
from src.data.download_images import download_images_from_drive
from src.preprocess.crop_yolo_person import run_yolo_person
from src.analysis.analyze_clip import analyze_clip_similarity_single
from src.visualize.match_rate import plot_match_rate_comparison
from src.analysis.combine_moods import combine_person_background
from src.preprocess.crop_yolo_background import run_yolo_background

IMAGE_DIR = 'data/rq1/rq1_images'
YOLO_PERSON_DIR = 'data/rq1/rq1_person'
YOLO_BACKGROUND_DIR = 'data/rq1/rq1_background'
YOLO_COMBINED_CSV = 'results/rq1/rq1_combined.csv'
YOLO_PERSON_CSV = 'results/rq1/rq1_person.csv'
YOLO_BACKGROUND_CSV = 'results/rq1/rq1_background.csv'
WHOLE_CSV = 'results/rq1/rq1_whole_clip.csv'
PLOT_PATH = 'results/rq1/match_rate_comparison.png'

DRIVE_FOLDER_ID = '14ZmYrfB8Uu9IiZzGZeLZ_QEHFTVdJd9y'

def main():
    baseline_model, baseline_processor = load_baseline_model()

    os.makedirs('data/rq1', exist_ok=True)
    download_images_from_drive(DRIVE_FOLDER_ID, IMAGE_DIR)

    run_yolo_person(IMAGE_DIR, YOLO_PERSON_DIR)
    run_yolo_background(IMAGE_DIR, YOLO_BACKGROUND_DIR)

    print("[3/4] Running CLIP analysis on masked images...")
    os.makedirs("results/rq1", exist_ok=True)
    analyze_clip_similarity_single(YOLO_PERSON_DIR, YOLO_PERSON_CSV, baseline_model, baseline_processor)
    analyze_clip_similarity_single(YOLO_BACKGROUND_DIR, YOLO_BACKGROUND_CSV, baseline_model, baseline_processor)

    print("[4/4] Combining results with weighted average...")
    person_df = pd.read_csv(YOLO_PERSON_CSV)
    background_df= pd.read_csv(YOLO_BACKGROUND_CSV)

    combined_df, _, match_rate = combine_person_background(person_df, background_df, weight_person=0.8)

    os.makedirs(os.path.dirname(YOLO_COMBINED_CSV), exist_ok=True)
    combined_df.to_csv(YOLO_COMBINED_CSV, index=False)

    with open(YOLO_COMBINED_CSV, "a") as f:
        f.write(f"\nmatch_rate,{match_rate:.4f}\n")

    print(f"[✔] Saved RQ2 combined result to {YOLO_COMBINED_CSV}")

    analyze_clip_similarity_single(IMAGE_DIR, WHOLE_CSV, baseline_model, baseline_processor)

    plot_match_rate_comparison(
        {
            "Combined (Person+BG)": YOLO_COMBINED_CSV,
            "Whole Image": WHOLE_CSV
        },
        PLOT_PATH
    )

if __name__ == "__main__":
    main()
