import torch, os
import torch.nn as nn

# =========================
# Config
# =========================
# Train Settings
num_classes     = 2
num_epochs      = 2
learning_rate   = 1e-4
optimizer_name  = "Adam"
criterion       = nn.CrossEntropyLoss()
frames_per_clip = 32

# Paths
current_directory   = "/home/smartan/Desktop/SlowfastTrainer/Models/Testing_2Classes_Cam10718"
model_save_path     = os.path.join(current_directory, "Testing_2Classes_Cam10718.pt")
model_save_arch_path= os.path.join(current_directory, "architecture.pt")
log_path            = os.path.join(current_directory, "SlowFast_training_log.txt")

# Data
train_datapath = "/home/smartan/Desktop/SlowfastTrainer/DatasetIdle_16Classes_Cam107-18/train"
val_datapath   = "/home/smartan/Desktop/SlowfastTrainer/DatasetIdle_16Classes_Cam107-18/val"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# Transforms (video-aware)
# =========================
try:
    print("_transforms_video is available")
    from torchvision.transforms._transforms_video import ResizeVideo, NormalizeVideo
    from torchvision.transforms import Compose
    transform = Compose([
        ResizeVideo((224, 224)),                               # (C,T,H,W)
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
except Exception:
    print("Fallback _transforms_video not available")
    # Fallback if _transforms_video not available (kept for compatibility)
    from torchvision.transforms import Compose, Resize
    from torchvision.transforms._transforms_video import NormalizeVideo
    transform = Compose([
        Resize((224, 224)),                                    # Works for some torchvision versions on (C,T,H,W)
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
