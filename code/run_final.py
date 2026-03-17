import os
import csv
import instaloader
import pandas as pd
import json
import requests
from dotenv import load_dotenv
from src.analysis.analyze_clip import analyze_clip_from_file
from tqdm import tqdm
from src.visualize.recommendation_viz import plot_mood_distribution

load_dotenv()

# Set your credential from .env
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
if not LASTFM_API_KEY:
    raise ValueError("LASTFM_API_KEY is not set. Add it to your .env file.")

# Mapping from moods to keywords
mood_TO_KEYWORDS = {
    "cozy": ["warm", "soft", "intimate", "relaxing"],
    "delicious": ["sweet", "tasty", "flavorful", "appetizing"],
    "excited": ["upbeat", "energetic", "fast", "fun"],
    "gloomy": ["dark", "sad", "melancholic", "slow"],
    "happy": ["bright", "cheerful", "joyful", "playful"],
    "lovely": ["romantic", "gentle", "sweet", "affectionate"],
    "ordinary": ["neutral", "simple", "casual", "calm"],
    "peaceful": ["calm", "instrumental", "quiet", "soothing"],
    "sensitive": ["moodal", "touching", "soft", "heartfelt"],
}

# Download Instagram post images
def download_images_from_post(post_url, download_dir='data/instagram_samples'):
    if os.path.exists(download_dir):
        for f in os.listdir(download_dir):
            os.remove(os.path.join(download_dir, f))
    os.makedirs(download_dir, exist_ok=True)
    loader = instaloader.Instaloader(download_pictures=True, save_metadata=False, download_videos=False)
    shortcode = post_url.strip("/").split("/")[-1]
    post = instaloader.Post.from_shortcode(loader.context, shortcode)
    loader.download_post(post, target=os.path.basename(download_dir))

# Recommend Spotify tracks based on mood
def recommend_tracks(mood_list, max_total=5, max_per_mood=2):
    recommended = []
    seen_tracks = set()

    for mood in tqdm(mood_list, desc="Recommending tracks"):
        keywords = mood_TO_KEYWORDS.get(mood, [mood])
        for keyword in keywords:
            try:
                response = requests.get(
                    f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={keyword}&api_key={LASTFM_API_KEY}&format=json&limit={max_per_mood}"
                )
                tracks = response.json().get('tracks', {}).get('track', [])
                for track in tracks:
                    name = track.get('name', 'N/A')
                    artist = track.get('artist', {}).get('name', 'N/A')
                    url = track.get('url', 'N/A')
                    track_key = f"{artist}-{name}"
                    if track_key in seen_tracks:
                        continue
                    seen_tracks.add(track_key)
                    recommended.append({
                        'mood': mood,
                        'tags': keyword,
                        'track_name': name,
                        'artist': artist,
                        'url': url
                    })
            except Exception as e:
                print(f"[ERROR] Last.fm tag fetch failed for '{keyword}': {e}")

    print(f"{len(recommended)} tracks recommended.")
    return recommended[:max_total]

# Main execution
if __name__ == "__main__":
    # insta_post_url = 'https://www.instagram.com/p/DJn2oXYSTTk/'  # Example URL, replace with actual post URL
    final_result_dir = 'results/final'
    os.makedirs(final_result_dir, exist_ok=True)
    insta_post_url = input("Enter Instagram post URL: ").strip()
    download_dir = 'data/instagram_samples'
    base_dir = os.path.basename(download_dir)
    download_images_from_post(insta_post_url, base_dir)

    image_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    moods_detected = []

    for image_file in tqdm(image_files, desc="Analyzing images"):
        image_path = image_file
        result = analyze_clip_from_file(image_path)
        best_mood = result.get("best_mood")
        if best_mood:
            moods_detected.append(best_mood)

    # Suffle and Remove duplicates
    unique_moods = list(dict.fromkeys(moods_detected))
    print("Detected moods:", unique_moods)

    plot_mood_distribution(moods_detected, save_path=os.path.join(final_result_dir, "mood_distribution.png"))

    tracks = recommend_tracks(mood_list=moods_detected, max_total=10)

    with open(os.path.join(final_result_dir, 'insta_music_recommendations.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['mood', 'tags', 'track_name', 'artist', 'url'])
        writer.writeheader()
        for track in tracks:
            writer.writerow(track)
