import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo
from collections import deque
import time
import onnxruntime as ort
from check_RTSP_stream import extract_rtsp_cap  # Optional if using RTSP

# Parameters
frames_per_clip = 32
alpha = 4

# Preprocessing pipeline
transform = Compose([
    Resize((256, 256)),
    CenterCrop(224),
    NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

# Path to ONNX model
onnx_path = r"C:\Users\Admin\Desktop\Git\Exercise_classification\Readymades\Slowfast\Just_SlowfastModule\Models\60Epochs_Cam10718_15classes\60epochs_cam10718_15classes_best.onnx"
session = ort.InferenceSession(onnx_path)
input_names = [inp.name for inp in session.get_inputs()]

# Class mapping
class_to_idx = {
    'BicepsCurls': 0,
    'dumbbell_chest_press': 1,
    'dumbbell_incline_chest_press': 2,
    'dumbbell_lunges': 3,
    'dumbbell_reverse_flys': 4,
    'FrontRaises': 5,
    'HammerCurls': 6,
    'kb_gobletsquats': 7,
    'kb_ohpress': 8,
    'kb_swings': 9,
    'kettlebell_goodmorning': 10,
    'LateralRaise': 11,
    'seated_dumbbell_shoulderpess': 12,
    'singlearm_dumbbell_rows': 13,
    'UprightRows': 14
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Helper to pack SlowFast inputs
def pack_pathway_output(frames, alpha=4):
    fast_pathway = frames
    slow_indices = np.linspace(0, frames.shape[2] - 1, frames.shape[2] // alpha).astype(int)
    slow_pathway = frames[:, :, slow_indices, :, :]
    return [slow_pathway, fast_pathway]

# Webcam or RTSP input
cap = cv2.VideoCapture(0)
# cap = extract_rtsp_cap(show_cap=False)

frame_queue = deque(maxlen=frames_per_clip)
print("Starting ONNX real-time inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    frame_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    frame_queue.append(frame_tensor)

    if len(frame_queue) == frames_per_clip:
        clip = torch.stack(list(frame_queue))  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        clip = transform(clip)  # Normalize
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T, C, H, W)
        clip = clip.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        clip_np = clip.numpy().astype(np.float32)
        print("Clip Mean:", clip_np.mean(), "Std:", clip_np.std())

        # Pack inputs for SlowFast
        slow_np, fast_np = pack_pathway_output(clip_np, alpha=alpha)
        inputs = {
            input_names[0]: slow_np,
            input_names[1]: fast_np
        }

        # Inference timing
        start = time.perf_counter()
        outputs = session.run(None, inputs)
        end = time.perf_counter()

        pred = np.argmax(outputs[0], axis=1)[0]
        pred_class = idx_to_class[pred]

        print(f"Inference time: {(end - start)*1000:.2f} ms | FPS: {1/(end - start):.2f}")
        print("Prediction:", pred_class)

        # Display prediction
        cv2.putText(frame, f"Pred: {pred_class}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ONNX SlowFast Real-time Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
