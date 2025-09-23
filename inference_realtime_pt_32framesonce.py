import cv2
import numpy as np
import torch
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.transforms import Compose
import torch.nn as nn
from collections import deque
import time
from get_RTSP_stream import extract_rtsp_cap
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

plt.ion()

# ---------- Settings ----------
frames_per_clip = 32
transform = Compose([
    Resize((256, 256)),
    CenterCrop(224),
    NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- Load Model ----------
def load_model(checkpoint_path, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------- SlowFast pathway packing ----------
def pack_pathway_output(frames, alpha=4):
    fast_pathway = frames  # shape: [1, 3, 32, 224, 224]
    slow_indices = torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // alpha).long()
    slow_pathway = torch.index_select(fast_pathway, 2, slow_indices.to(device))
    return [slow_pathway, fast_pathway]


# ---------- Real-time Prediction ----------
def infer_live_video(model, class_names, cam_id=0):

    buffer = deque(maxlen=frames_per_clip)
    frame_counter = 0

    cap = extract_rtsp_cap()

    if cap is None:
        raise RuntimeError("Failed to open RTSP strea")
    
    pred_class = "Detecting..."  # <-- Initialize with a default

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb_frame)
        frame_counter += 1

        # Once we have enough frames
        if len(buffer) == frames_per_clip:  # e.g., 32
            frame_indices = np.linspace(0, frames_per_clip - 1, frames_per_clip).astype(int)
            sampled_frames = [buffer[i] for i in frame_indices]

            # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
            frames_np = np.stack(sampled_frames)  # (32, H, W, 3)
            frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # (32, 3, H, W)

            # Permute for transformation
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (3, 32, H, W)
            frames_tensor = transform(frames_tensor)
            frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, 32, 3, 224, 224)

            # Final permute to (B, C, T, H, W)
            frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)

            print(f"Frames_tensor: {frames_tensor.shape}") #Frames_tensor: torch.Size([1, 3, 32, 224, 224])
            inputs = pack_pathway_output(frames_tensor)

            with torch.no_grad():
                print(f"Input shape: {inputs[0].shape}")
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).item()
                pred_class = class_names[pred]

                print(f"[{time.strftime('%H:%M:%S')}] Prediction: {pred_class}")

            # Reset buffer for next 32 frames
            buffer.clear() 

        # Show latest frame with prediction overlay
        cv2.putText(frame, f"Prediction: {pred_class}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow("Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    checkpoint_path = r"/SlowfastTrainer/Models/Idle_E60_C16_Cam10718/Idle_E60_C16_Cam10718.pt"

    class_names = [
    'Bicep_Curls',
    'DB_Chest_Press',
    'DB_Incline_Chest_Press',
    'DB_Lunges',
    'DB_Reverse_Flys',
    'Front_Raises',
    'Hammer_Curls',
    'KB_Goblet_Squats',
    'KB_Overhead_Press',
    'KB_Swings',
    'KB_Goodmorning',
    'Lateral_Raise',
    'Seated_DB_Shoulder_Press',
    'Singlearm_DB_Rows',
    'Upright_Rows',
    'Idle'
    ]
    
     # Fill in exactly as in idx_to_class.values()
    model = load_model(checkpoint_path, num_classes=len(class_names))
    infer_live_video(model, class_names)
