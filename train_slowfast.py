#!/usr/bin/env python3
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *

# =========================
# Dataset
# =========================
from decord import VideoReader, cpu

class YourVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._build_index()

    def _build_index(self):
        print("########### BUILD INDEX TRACKING ###########")
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        print(f" |Classes: {classes}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f" |Class_to_idx: {self.class_to_idx}")
        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            print(f" |Class_directory: {cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])
        print(f" |Num videos: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)

            # Frame indices
            if total_frames < self.frames_per_clip:
                # pad by repeating last frame (simple + robust)
                base = np.linspace(0, total_frames - 1, total_frames).astype(int)
                pad = self.frames_per_clip - total_frames
                frame_indices = np.concatenate([base, np.full((pad,), base[-1], dtype=int)])
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)

            frames = vr.get_batch(frame_indices).asnumpy()          # (T,H,W,C)

            # Fix for grayscale videos
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] != 3:
                raise ValueError(f"Unsupported channel count: {frames.shape[-1]} in video {path}")

            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0   # (C,T,H,W)

            if self.transform:
                frames = self.transform(frames)                                    # keep (C,T,H,W)

            # return (T,C,H,W) to match your later permute call
            frames = frames.permute(1, 0, 2, 3).contiguous()                       # (T,C,H,W)

            return frames, label

        except Exception as e:
            print(f"Failed to load video: {path}\nError: {e}")
            # try next video (avoid infinite recursion if dataset has 0 length)
            return self.__getitem__((idx + 1) % len(self))

# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    preds, targets = [], []
    val_loss = 0.0

    for videos, labels in val_loader:
        # videos: (B,T,C,H,W) -> (B,C,T,H,W)
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        labels = torch.as_tensor(labels, device=device)

        # SlowFast dual pathway
        T = videos.shape[2]
        n_slow = max(1, T // 4)
        slow_indices = torch.linspace(0, T - 1, n_slow).long().to(device)
        slow_pathway = videos.index_select(2, slow_indices)
        inputs = [slow_pathway, videos]

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        preds.extend(predicted.detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    acc = correct / total if total else 0.0
    precision = precision_score(targets, preds, average='macro', zero_division=0)
    recall    = recall_score(targets, preds, average='macro', zero_division=0)
    f1        = f1_score(targets, preds, average='macro', zero_division=0)
    return val_loss / max(1, len(val_loader)), acc, precision, recall, f1


# =========================
# Model
# =========================
os.makedirs(current_directory, exist_ok=True)

# Base model
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

# Replace final head
if hasattr(model.blocks[-1], "proj") and isinstance(model.blocks[-1].proj, nn.Linear):
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
else:
    # Fallback: try common location
    raise RuntimeError("Could not find final projection layer to replace.")

model = model.to(device)
print(f"Model ready for fine-tuning with {num_classes} output classes.")

# Optimizer
if optimizer_name == "Adam":
    print(f"Activating Optimizer {optimizer_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# Datasets & Loaders
train_dataset = YourVideoDataset(train_datapath, transform=transform, frames_per_clip=frames_per_clip)
val_dataset   = YourVideoDataset(val_datapath,   transform=transform, frames_per_clip=frames_per_clip)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

# =========================
# Sanity check: one backward to confirm grad flow
# =========================
model.train()
with torch.enable_grad():
    videos0, labels0 = next(iter(train_loader))
    videos0 = videos0.permute(0, 2, 1, 3, 4).to(device)   # (B,C,T,H,W)
    labels0 = torch.as_tensor(labels0, device=device)
    T0 = videos0.shape[2]
    n_slow0 = max(1, T0 // 4)
    slow_idx0 = torch.linspace(0, T0 - 1, n_slow0).long().to(device)
    slow0 = videos0.index_select(2, slow_idx0)
    out0 = model([slow0, videos0])
    loss0 = criterion(out0, labels0)
    loss0.backward()

    head = getattr(model.blocks[-1], "proj", None)
    if head is None or head.weight.grad is None:
        raise RuntimeError("Sanity check failed: no gradient on final classification head.")
    print(f"[Sanity] loss={float(loss0):.6f}, head||grad||={head.weight.grad.data.norm(2).item():.6e}")
    model.zero_grad(set_to_none=True)

# =========================
# Training Loop
# =========================
best_loss = float('inf')
start_time = time.time()

for epoch in range(num_epochs):
    print(f"\n------------------------------- EPOCH {epoch+1} -------------------------------")
    epoch_start_time = time.time()
    model.train()

    running_loss = 0.0
    total_grad_norm = 0.0

    # reset per-epoch training metrics
    all_train_preds = []
    all_train_labels = []

    for videos, labels in train_loader:
        # videos: (B,T,C,H,W) -> (B,C,T,H,W)
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        labels = torch.as_tensor(labels, device=device)

        # SlowFast dual pathway
        T = videos.shape[2]
        n_slow = max(1, T // 4)
        slow_indices = torch.linspace(0, T - 1, n_slow).long().to(device)
        slow_pathway = videos.index_select(2, slow_indices)
        inputs = [slow_pathway, videos]

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Per-parameter grad norm check (avoid misleading sum==0)
        any_finite = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach()
                gnorm = g.data.norm(2).item()
                if np.isfinite(gnorm) and gnorm > 0:
                    any_finite = True

        if not any_finite:
            print("[grad check] All parameter grad norms are zero/NaN this step.")

        # Total grad norm
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                pn = p.grad.data.norm(2).item()
                total_norm_sq += pn * pn
        total_grad_norm = float(total_norm_sq ** 0.5)

        optimizer.step()
        running_loss += float(loss.item())

        # Collect epoch training metrics
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_train_preds.extend(preds.tolist())
        all_train_labels.extend(labels.detach().cpu().numpy().tolist())

    # Epoch metrics
    avg_train_loss = running_loss / max(1, len(train_loader))
    train_accuracy = accuracy_score(all_train_labels, all_train_preds) if all_train_labels else 0.0
    train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0) if all_train_labels else 0.0
    train_recall    = recall_score(all_train_labels, all_train_preds,   average='weighted', zero_division=0) if all_train_labels else 0.0
    train_f1        = f1_score(all_train_labels, all_train_preds,       average='weighted', zero_division=0) if all_train_labels else 0.0

    # Validation
    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = validate(model, val_loader, criterion, device)

    # Save best
    saved = False
    if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), model_save_path)
        torch.save(model, model_save_arch_path)
        saved = True

    # Logs
    print(f"Train Loss: {avg_train_loss:.6f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
    print(f"Eval  Loss: {eval_loss:.6f}, Acc: {eval_accuracy:.4f}, Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}")
    print(f"Grad Norm (last step): {total_grad_norm:.6f}, LR: {optimizer.param_groups[0]['lr']:.10f}")
    if saved:
        print(f"Model saved to: {os.path.abspath(model_save_path)}")
    print(f"Epoch Time Taken: {(time.time() - epoch_start_time):.2f} sec")

    with open(log_path, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Train Loss: {avg_train_loss:.6f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n")
        f.write(f"Eval Loss: {eval_loss:.6f}, Accuracy: {eval_accuracy:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, F1: {eval_f1:.4f}\n")
        f.write(f"Grad Norm: {total_grad_norm:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']:.10f}\n")
        if saved:
            f.write(f"Model saved to: {os.path.abspath(model_save_path)}\n")
        f.write(f"Epoch Time Taken: {(time.time() - epoch_start_time):.2f} sec\n")
        f.write("----------------------------------------------------------\n")

# Total time
print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes.")
