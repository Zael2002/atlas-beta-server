
"""
Atlas Beta - Image-based Signal Generator (starter)
File: atlas_image_beta.py
Description:
    - Receives chart images (prints/screenshots) and produces a signal (CALL / PUT / NEUTRAL)
      together with an annotated image showing Atlas' reasoning hints.
    - Two analysis modes:
        1) Heuristic mode: quick rule-based extraction from image (no ML model required)
        2) Model mode: transfer-learning CNN classifier (requires a trained model file `model.pth`)
    - This starter is designed to get you testing fast. For highest accuracy, you should:
        * Build/label a dataset of screenshots -> signals
        * Train the CNN (training loop included below) with augmentation
        * Iterate ensemble: combine heuristics + CNN + meta-classifier
Requirements (install with pip):
    pip install opencv-python-headless pillow numpy matplotlib torch torchvision tqdm scikit-learn
Notes:
    - This script does NOT perform live trading; it only analyzes images and emits signals.
    - To reach "the highest accuracy", you must collect labeled image data and train the model.
Author: Atlas Team (starter script)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from datetime import datetime

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"   # if exists, model mode is used
OUTPUT_DIR = "atlas_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image sizes for model input
IMG_SIZE = 320

# Simple thresholds for heuristic
HEUR_RISE_SLOPE_THRESH = 0.0015
HEUR_VOL_BREAK_THRESH = 1.5  # relative volume threshold (approximate)

# ----------------------------

# ---- Utilities ----
def load_image(path, target_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size))
    return img

def save_annotated_image(orig_path, annotated_img_pil, signal_text):
    base = os.path.basename(orig_path)
    name, ext = os.path.splitext(base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{OUTPUT_DIR}/{name}_{signal_text}_{ts}.png"
    annotated_img_pil.save(out_name)
    return out_name

# ---- Heuristic Analyzer (quick fallback) ----
def heuristic_analyze(image_path):
    """
    Very simple image-based heuristic:
    - Convert to grayscale
    - Crop center region (where candles usually appear)
    - Detect the dominant slope of the brightest/darkest line (approx trend)
    - Detect large bright vertical contours (possible big candles -> momentum)
    Returns: dict with signal and reasons
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"signal":"NEUTRAL", "confidence":0.0, "reasons":["unable_to_read_image"]}
    h, w = img.shape[:2]
    # center crop
    cy, cx = h//2, w//2
    crop = img[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # smooth and Canny for structure
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Hough lines -> estimate slope
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
    slopes = []
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2==x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
    # candle-like vertical contours as momentum proxy
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_count = 0
    areas = []
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        if hh > ww * 2 and hh > 10:
            vertical_count += 1
            areas.append(ww*hh)
    avg_slope = np.median(slopes) if len(slopes)>0 else 0.0
    avg_area = np.median(areas) if len(areas)>0 else 0.0
    reasons = []
    # interpret slope (image Y coordinate increases downward -> negative slope ~ upward trend visually)
    # small heuristic sign inversion to make sense
    trend_score = -avg_slope  # positive => upward trend
    if trend_score > HEUR_RISE_SLOPE_THRESH:
        reasons.append("upward_trend_detected")
    elif trend_score < -HEUR_RISE_SLOPE_THRESH:
        reasons.append("downward_trend_detected")
    else:
        reasons.append("no_clear_trend")
    if vertical_count > 3:
        reasons.append("high_momentum_candles")
    # Simple decision rules
    if "upward_trend_detected" in reasons and "high_momentum_candles" in reasons:
        signal = "CALL"
        confidence = min(0.6 + min(0.3, avg_area/5000.0), 0.95)
    elif "downward_trend_detected" in reasons and "high_momentum_candles" in reasons:
        signal = "PUT"
        confidence = min(0.6 + min(0.3, avg_area/5000.0), 0.95)
    else:
        signal = "NEUTRAL"
        confidence = 0.35
    return {"signal": signal, "confidence": round(float(confidence),3), "reasons": reasons, "meta":{"avg_slope":float(avg_slope), "verticals":int(vertical_count)}}

# ---- Model (Transfer Learning) ----
class AtlasClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # use efficient backbone if available; fallback to resnet18
        try:
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            # replace classifier
            self.backbone.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, n_classes))
        except Exception:
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

# Preprocessing transforms for model
MODEL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_trained_model(path=MODEL_PATH, device=DEVICE):
    if not os.path.exists(path):
        return None
    model = AtlasClassifier(n_classes=3)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

LABELS = ["CALL","PUT","NEUTRAL"]

def predict_with_model(model, pil_image):
    img_t = MODEL_TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"signal": LABELS[idx], "confidence": float(round(float(probs[idx]),3)), "probs": probs.tolist()}

# ---- Annotation ----
def annotate_image(image_path, analysis):
    pil = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    w,h = pil.size
    # draw top banner
    banner_h = int(h*0.12)
    banner = Image.new("RGBA", (w,banner_h), (0,0,0,190))
    pil.paste(banner, (0,0), banner)
    # write text
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, banner_h//6))
    except Exception:
        font = ImageFont.load_default()
    txt = f"Atlas -> SIGNAL: {analysis['signal']}   CONF: {analysis['confidence']}"
    draw.text((10,10), txt, font=font, fill=(255,255,255,255))
    # reasons
    reasons = ', '.join(analysis.get("reasons", []))
    draw.text((10, 10 + banner_h//2), f"Reasons: {reasons}", font=font, fill=(220,220,220,230))
    # draw small meta values
    meta = analysis.get("meta", {})
    draw.text((w-220,10), f"meta: {meta}", font=font, fill=(200,200,200,220))
    # return PIL image RGBA
    return pil.convert("RGB")

# ---- Main analyze function ----
def analyze_image(path):
    # try model first
    model = load_trained_model()
    pil = load_image(path)
    if model is not None:
        try:
            model_res = predict_with_model(model, pil)
            # also run heuristics for ensemble signal
            heur = heuristic_analyze(path)
            # simple ensemble: if both agree -> increase confidence; else prefer model but show both reasons
            if model_res['signal'] == heur['signal'] and model_res['signal'] != "NEUTRAL":
                conf = min(0.98, model_res['confidence'] + heur['confidence']*0.25)
                out = {"signal": model_res['signal'], "confidence": round(conf,3), "reasons":[f"model:{model_res['signal']}", f"heur:{heur['signal']}"], "meta":{"model_probs":model_res.get("probs"), "heur_meta":heur.get("meta")}}
            else:
                # prefer model
                out = {"signal": model_res['signal'], "confidence": model_res['confidence'], "reasons":[f"model:{model_res['signal']}", f"heur:{heur['signal']}"], "meta":{"model_probs":model_res.get("probs"), "heur_meta":heur.get("meta")}}
        except Exception as e:
            # fallback
            out = heuristic_analyze(path)
            out['reasons'].append("model_failed")
    else:
        out = heuristic_analyze(path)
    # produce annotated image
    annotated = annotate_image(path, out)
    saved = save_annotated_image(path, annotated, out['signal'])
    out['annotated_image'] = saved
    return out

# ---- Training helper (skeleton) ----
def train_model(data_dir, epochs=8, batch_size=16, lr=1e-4):
    """
    data_dir structure:
        data_dir/CALL/*.jpg
        data_dir/PUT/*.jpg
        data_dir/NEUTRAL/*.jpg
    This is a skeleton training loop using torchvision ImageFolder.
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    dataset = ImageFolder(data_dir, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = AtlasClassifier(n_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {acc:.4f}")
        # save checkpoint each epoch
        torch.save(model.state_dict(), MODEL_PATH)
    print("Training finished. Model saved to", MODEL_PATH)

# ---- CLI ----
def print_banner():
    print("Atlas Beta - Image-based Signal Generator")
    print("Modes: model (if model.pth exists) + heuristic fallback")
    print("Usage examples:")
    print("  python atlas_image_beta.py analyze path/to/image.png")
    print("  python atlas_image_beta.py train path/to/labeled_dataset/")
    print("")

def main():
    import sys
    if len(sys.argv) < 2:
        print_banner()
        return
    cmd = sys.argv[1].lower()
    if cmd == "analyze" and len(sys.argv)>=3:
        image_path = sys.argv[2]
        res = analyze_image(image_path)
        print("Result:", res['signal'], "conf:", res['confidence'])
        print("Annotated image saved to:", res['annotated_image'])
    elif cmd == "train" and len(sys.argv)>=3:
        data_dir = sys.argv[2]
        train_model(data_dir)
    else:
        print_banner()

if __name__ == "__main__":
    main()
