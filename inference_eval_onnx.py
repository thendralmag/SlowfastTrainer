import os
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo
import torch
import onnxruntime as ort


class YourVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=32):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._build_index()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _build_index(self):
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < self.frames_per_clip:
                frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
                pad = self.frames_per_clip - total_frames
                frame_indices = np.concatenate([frame_indices, [frame_indices[-1]] * pad])
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)

            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] != 3:
                raise ValueError(f"Unsupported channel count: {frames.shape[-1]} in video {path}")

            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

            if self.transform:
                frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
                frames = self.transform(frames)
                frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)

            return frames, label, path

        except Exception as e:
            print(f"Failed to load video: {path}\nError: {e}")
            return self.__getitem__((idx + 1) % len(self))


def pack_pathway_output(frames, alpha=4):
    """
    Convert (B, C, T, H, W) -> [slow, fast] for ONNX
    """
    fast_pathway = frames
    slow_indices = torch.linspace(0, fast_pathway.shape[2] - 1, fast_pathway.shape[2] // alpha).long()
    slow_pathway = torch.index_select(fast_pathway, 2, slow_indices.to(fast_pathway.device))
    return [slow_pathway.cpu().numpy(), fast_pathway.cpu().numpy()]


def run_inference(onnx_session, dataloader, idx_to_class):
    input_names = [inp.name for inp in onnx_session.get_inputs()]
    results = []

    for videos, labels, paths in dataloader:
        videos = videos.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        slow, fast = pack_pathway_output(videos)

        inputs = {input_names[0]: slow, input_names[1]: fast}
        outputs = onnx_session.run(None, inputs)
        preds = np.argmax(outputs[0], axis=1)

        for path, pred in zip(paths, preds):
            results.append((path, idx_to_class[pred]))

    return results


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the model.onnx")
    parser.add_argument("--data_dir", required=True, help="Path to class-wise .mp4 video folders")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    transform = Compose([
        Resize((256, 256)),
        CenterCrop(224),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])

    dataset = YourVideoDataset(args.data_dir, transform=transform, frames_per_clip=32)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    idx_to_class = dataset.idx_to_class
    print("idx_to_class", idx_to_class)

    ort_session = ort.InferenceSession(args.checkpoint)

    results = run_inference(ort_session, dataloader, idx_to_class)

    for path, pred_class in results:
        print(f"Video: {os.path.basename(path)} | Predicted Class: {pred_class}")
        with open("inference_results_onnx.txt", "a") as f:
            f.write(f"Video: {os.path.basename(path)} | Predicted Class: {pred_class}\n")
