import os
from src.models.load_models import load_baseline_model
from src.data.download_images import download_images_from_drive
from src.preprocess.crop_mtcnn_person import run_mtcnn_person
from src.preprocess.crop_yolo_person import run_yolo_person
from src.analysis.analyze_clip import analyze_clip_similarity_single
from src.visualize.match_rate import plot_match_rate_comparison

IMAGE_DIR = 'data/rq245/rq245_images'
YOLO_PERSON_DIR = 'data/rq245/rq245_yolo_person'
MTCNN_PERSON_DIR = 'data/rq245/rq245_mtconn_person'
YOLO_PERSON_CSV = 'results/rq245/rq245_yolo_person.csv'
MTCNN_PERSON_CSV = 'results/rq245/rq245_mtcnn_person.csv'
PLOT_PATH = "results/rq245/match_rate_comparison.png"

DRIVE_FOLDER_ID = "106CLtANP_dGKeCsyu9GlsX8kgjxFsSsX"

def main():
    baseline_model, baseline_processor = load_baseline_model()

    os.makedirs('data/rq245', exist_ok=True)
    download_images_from_drive(DRIVE_FOLDER_ID, IMAGE_DIR)    

    # YOLO crop
    run_yolo_person(IMAGE_DIR, YOLO_PERSON_DIR)

    # MTCNN crop
    run_mtcnn_person(IMAGE_DIR, MTCNN_PERSON_DIR)

    print("[3/4] Running CLIP analysis on YOLO, PERSON images...")
    os.makedirs('results/rq245', exist_ok=True)
    analyze_clip_similarity_single(YOLO_PERSON_DIR, YOLO_PERSON_CSV, baseline_model, baseline_processor)
    analyze_clip_similarity_single(MTCNN_PERSON_DIR, MTCNN_PERSON_CSV, baseline_model, baseline_processor)

    print("[4/4] 결과 비교")
    # Compare
    plot_match_rate_comparison(
        {
            "YOLO": YOLO_PERSON_CSV,
            "MTCNN": MTCNN_PERSON_CSV
        },
        PLOT_PATH
    )

if __name__ == "__main__":
    main()