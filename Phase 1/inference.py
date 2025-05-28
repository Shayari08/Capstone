import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
from collections import deque, Counter
from PIL import Image

# ---- Load Class Labels ----
def load_gloss_labels(gloss_file):
    with open(gloss_file, "r") as f:
        return [line.strip() for line in f.readlines()]

# ---- Load Trained Model ----
class AttnNet(torch.nn.Module):
    def __init__(self, num_classes=100):
        super(AttnNet, self).__init__()
        self.feature_extractor = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.temporal_conv = torch.nn.Conv1d(64, 256, kernel_size=3, padding=1)
        self.attention = torch.nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        x = self.feature_extractor(frames)
        x = F.relu(x)
        x = x.view(B, T, 64, H, W)
        x = torch.mean(x, dim=(3, 4))  # Average pooling
        x = self.temporal_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        return self.fc(attn_out)  # Return per-frame logits

# ---- Video Preprocessing ----
def preprocess_video(video_path, img_size=(128, 128), save_dir=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        transformed = transform(pil_image)
        frames.append(transformed)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{frame_idx:04d}.jpg")
            pil_image.resize(img_size).save(save_path)
        frame_idx += 1

    cap.release()
    frames = torch.stack(frames) if frames else None
    return frames.unsqueeze(0) if frames is not None else None

# ---- Smoothing Function ----
def smooth_predictions(predictions, confidence_scores, threshold=0.6, window_size=5):
    smoothed = []
    buffer = deque(maxlen=window_size)
    prev = None

    for gloss, conf in zip(predictions, confidence_scores):
        buffer.append(gloss)
        most_common, freq = Counter(buffer).most_common(1)[0]
        if conf < threshold and prev is not None:
            smoothed.append(prev)
        else:
            smoothed.append(most_common)
            prev = most_common

    return smoothed

# ---- Collapse Repeats ----
def collapse_repeats(sequence):
    collapsed = []
    prev = None
    for gloss in sequence:
        if gloss != prev:
            collapsed.append(gloss)
            prev = gloss
    return collapsed

# ---- Run Inference ----
def predict_gloss_sequence(model, video_path, gloss_labels, device, save_frame_dir=None):
    model.eval()
    model.to(device)

    frames = preprocess_video(video_path, save_dir=save_frame_dir)
    if frames is None:
        print("Error: Unable to process video frames.")
        return []

    frames = frames.to(device)
    with torch.no_grad():
        outputs = model(frames)  # Shape: [1, T, num_classes]
        probabilities = F.softmax(outputs, dim=2)
        confidence_scores, predictions = torch.max(probabilities, dim=2)

    predicted_glosses = [gloss_labels[idx] for idx in predictions[0].cpu().numpy()]
    confidence_scores = confidence_scores[0].cpu().numpy()

    smoothed_glosses = smooth_predictions(predicted_glosses, confidence_scores)
    collapsed_glosses = collapse_repeats(smoothed_glosses)
    return collapsed_glosses

# ---- Main Execution ----
if __name__ == "__main__":
    model_path = "C:\\Users\\shaya\\OneDrive\\Desktop\\project\\attnnet_model.pth"
    video_path = "C:\\Users\\shaya\\Downloads\ASL have a good day.mp4"
    gloss_file = "C:\Users\shaya\OneDrive\Desktop\project\wlasl_class_list.txt"

    print("Extracting frames...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gloss_labels = load_gloss_labels(gloss_file)
    print("Gloss dictionary generated")

    print("Loading model...")
    model = AttnNet(num_classes=300)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except RuntimeError as e:
        print(f"Error loading model: {e}")

    os.makedirs("output", exist_ok=True)
    gloss_sequence = predict_gloss_sequence(model, video_path, gloss_labels, device, save_frame_dir="output/frames")

    with open("output/predicted_glosses.txt", "w") as f:
        f.write(" ".join(gloss_sequence))

    print("Everything saved in: output")
