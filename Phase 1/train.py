import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from multiprocessing import freeze_support

# ----------------- Configuration ----------------- #
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "D:\\lilacs\\preprocessed_frames"
METADATA_PATH = "D:\\lilacs\\dataset\\nslt_300.json"
NUM_CLASSES = 300
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4

# ----------------- Dataset ----------------- #
def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

class CSLRDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.video_ids = list(metadata.keys())
        self.transform = transform

    def __len__(self):
        return len(self.video_ids)

    def load_frames_and_cams(self, video_id):
        frame_dir = os.path.join(DATASET_PATH, video_id)
        if not os.path.exists(frame_dir):
            return None, None

        frames, cams = [], []
        for i in range(1, 10000):
            frame_path = os.path.join(frame_dir, f"{i:04d}.jpg")
            cam_path = os.path.join(frame_dir, f"{i:04d}.npy")

            if not os.path.exists(frame_path) or not os.path.exists(cam_path):
                break

            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame) if self.transform else torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
            cam = torch.tensor(np.load(cam_path), dtype=torch.float32)
            frames.append(frame)
            cams.append(cam)

        return frames, cams

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = self.metadata[video_id]['action'][0]
        frames, cams = self.load_frames_and_cams(video_id)
        if frames is None:
            return None
        return torch.stack(frames), torch.stack(cams), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    frames, cams, labels = zip(*batch)
    frames = pad_sequence(frames, batch_first=True, padding_value=0)
    cams = pad_sequence(cams, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return frames, cams, labels

# ----------------- Model ----------------- #
class AttnNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.conv1d = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.cam_proj = nn.Linear(1, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, frames, cams):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        feats = self.resnet(frames).view(B, T, 512).permute(0, 2, 1)
        feats = self.conv1d(feats).permute(0, 2, 1)
        feats = self.layer_norm(feats)

        if cams.dim() == 6 and cams.shape[2:] == (1, 512, 7, 7):
            cams = cams.mean(dim=[2, 3, 4, 5])
            cams = cams.unsqueeze(-1)
        elif cams.dim() == 5 and cams.shape[3:] == (7, 7):
            cams = cams.mean(dim=[3, 4])
        elif cams.dim() == 3 and cams.shape[2] == 1:
            pass
        else:
            raise ValueError(f"[ERROR] Unexpected CAM shape: {cams.shape}")

        cam_weights = torch.sigmoid(self.cam_proj(cams))
        feats = feats * cam_weights
        attn_out, _ = self.attn(feats, feats, feats)
        feats = feats + attn_out
        out = self.fc(self.dropout(feats.mean(dim=1)))
        return out

# ----------------- Training and Eval ----------------- #
def train(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        if batch is None:
            continue
        frames, cams, labels = batch
        frames, cams, labels = frames.to(DEVICE), cams.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            outputs = model(frames, cams)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None:
                continue
            frames, cams, labels = batch
            frames, cams, labels = frames.to(DEVICE), cams.to(DEVICE), labels.to(DEVICE)
            outputs = model(frames, cams)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

# ----------------- Main Execution ----------------- #
if __name__ == '__main__':
    freeze_support()
    metadata = load_metadata()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CSLRDataset(metadata, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)

    model = AttnNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, dataloader, criterion, optimizer, scaler)
        acc, prec, rec, f1 = evaluate(model, dataloader)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    torch.save(model.state_dict(), f"attnnet.pth")
