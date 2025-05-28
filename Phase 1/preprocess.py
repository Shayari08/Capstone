import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from glob import glob
from tqdm import tqdm
import json

# ================== Paths ==================
VIDEO_DIR = "D:\\lilacs\\dataset\\videos" # Path to raw videos
OUTPUT_DIR = "D:\\lilacs\\preprocessed_frames"  # Store both frames & CAMs
METADATA_FILE = "D:\\lilacs\\dataset\\nslt_300.json"  # Metadata file

# ================== Load Pretrained Model for CAM Extraction ==================
def load_resnet():
    """Load ResNet model without the final classification layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-2])  # Remove last FC layer
    model.eval()
    return model

# ================== Frame Extraction ==================
def extract_frames(video_path, output_folder):
    """Extract frames from a video and save them as .jpg images."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frames.append(frame_filename)
        frame_count += 1

    cap.release()
    return saved_frames  # Return list of saved frames

# ================== CAM Extraction ==================
def extract_cams(frames_folder, model):
    """Extract Class Activation Maps (CAMs) for frames using ResNet."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_paths = sorted(glob(os.path.join(frames_folder, "*.jpg")))
    cam_outputs = []

    for frame_path in tqdm(frame_paths, desc="Extracting CAMs"):
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0)

        with torch.no_grad():
            cam_output = model(frame).cpu().numpy()

        cam_filename = frame_path.replace(".jpg", ".npy")
        np.save(cam_filename, cam_output)
        cam_outputs.append(cam_filename)

    return cam_outputs  # Return list of saved CAM files

# ================== Cleanup Empty Folders ==================
def cleanup_empty_folders(base_folder):
    """Remove any empty folders to ensure clean preprocessing."""
    for root, dirs, files in os.walk(base_folder, topdown=False):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            if not os.listdir(folder_path):  # Check if folder is empty
                os.rmdir(folder_path)

# ================== Main Processing ==================
if __name__ == "__main__":
    # Load metadata file
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    video_ids = list(metadata.keys())

    # Load ResNet model
    resnet_model = load_resnet()

    for video_id in tqdm(video_ids, desc="Processing Videos"):
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        output_folder = os.path.join(OUTPUT_DIR, video_id)

        if not os.path.exists(video_path):
            print(f"⚠️ Warning: Video {video_id} not found, skipping...")
            continue

        saved_frames = extract_frames(video_path, output_folder)
        if not saved_frames:
            print(f"⚠️ Warning: No frames extracted for {video_id}, skipping CAM extraction.")
            continue  # Skip CAM processing if no frames exist

        extract_cams(output_folder, resnet_model)

    # Cleanup any empty folders
    cleanup_empty_folders(OUTPUT_DIR)
    print("✅ Preprocessing complete. All frames and CAMs are stored in:", OUTPUT_DIR)
