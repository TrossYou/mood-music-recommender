import torch
import os
from transformers import CLIPModel, CLIPProcessor

# 디바이스 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_baseline_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model.eval(), processor


def load_finetuned_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Finetuned model file not found: {model_path}")
   
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
