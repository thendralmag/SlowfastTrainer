import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torch.utils.data import DataLoader, Dataset
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import argparse
import torchvision.transforms.functional as TF
from torchvision.transforms._transforms_video import NormalizeVideo
import imageio, cv2




class YourVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=32):
        print(f"3. Initializing VideoDataset")
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._build_index()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _build_index(self):

        print("########### BUILD INDEX TRACKING ###########")

        classes = sorted(os.listdir(self.root_dir))

        print(f"4. Sorted Classes: {classes}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"5. Class_to_idx: {self.class_to_idx}")
        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            # print(f" |Class_directory: {cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

        print(f"6. Appended Video_paths length: {len(self.video_paths)}")
        print(f"7. Appended Labels: {self.labels}")

    def __len__(self):
        print(f" |Length of VideoPaths: {len(self.video_paths)}")
        return len(self.video_paths)


    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        print(f"8. Video Paths in __getitem__ {path}")
        print(f"9. Labels in __getitem__ {label}")

        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)
            # print("#####################################")
            print("10. Total Frames: ", total_frames)


            """BELOW IF CONDITION: Handles short videos by duplicating last frame to match the required number of frames."""
            """padding short videos by repeating the last frame can reduce the effectiveness and learning quality of the model"""
            """Downsides of this method: Redundant Information, Bias Toward Last Frame, Temporal Dynamics Are Lost"""
            if total_frames < self.frames_per_clip:
                print("Total frames < frames oer clip")
                print(f"11. Total frames is less than ", self.frames_per_clip)
                frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
                print(f"12. Line Spaced using ({0, total_frames - 1, total_frames})")
                pad = self.frames_per_clip - total_frames
                print(f"13. Pad: {pad}")
                frame_indices = np.concatenate([frame_indices, [frame_indices[-1]] * pad])
                print(f"14. frame_indices: {frame_indices}")


                """Better Alternatives: Loop the video content"""
                # # Repeat the sequence to fill the clip length
                # repeat_factor = int(np.ceil(self.frames_per_clip / total_frames))
                # frame_indices = np.tile(np.arange(total_frames), repeat_factor)[:self.frames_per_clip]

            else:
                print("Total frame >= frames per clip")
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
                print(f"11. Line Spaced using ({0, total_frames - 1, self.frames_per_clip})")
                selected_frames = [vr[i].asnumpy() for i in frame_indices]
                # print(f"12. selected Frames: {selected_frames}")
                print(f"13. frame_indices: {frame_indices}")
                output_path = f"{os.path.basename(path).split('.')[0]}_sampled_output.mp4"
                output_path = os.path.join("/home/smartan/Desktop/Just_SlowfastModule/debugg", output_path)  # customize output folder
                os.makedirs("/home/smartan/Desktop/Just_SlowfastModule/debugg", exist_ok=True)
                writer = imageio.get_writer(output_path, fps=32)
                print("14. NA")

                for frame in selected_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                writer.close()

            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
            print(f"15. Finalized Frames to process further: {frames.shape}")


            # Fix for grayscale (C=1) videos
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] != 3:
                raise ValueError(f"Unsupported channel count: {frames.shape[-1]} in video {path}")
            
            print("16. Initial Frame Shape", frames.shape)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            print("17. First Permute Frame Shape (0, 3, 1, 2)", frames.shape)

            if self.transform:
                print(f"18.Transformation Applying: {self.transform}")
                frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
                print("19. Second Permute Frame Shape (1, 0, 2, 3)", frames.shape)
                frames = self.transform(frames)
                print("20. Transformation Applied -----> Frame shape Now:", frames.shape)
                frames = frames.permute(1, 0, 2, 3)  # Back to (T, C, H, W)
                print(f"21. Third permute Frame Shape (1, 0, 2, 3)", frames.shape)

            print(f"22. Returning Frames: {frames.shape}, label: {label}, path: {path}")
            return frames, label, path

        except Exception as e:
            print(f"Failed to load video: {path}\nError: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next video

def pack_pathway_output(frames, alpha=4):
    """
    Create inputs for SlowFast model from a clip.

    Args:
        frames (Tensor): shape (T, C, H, W)
        alpha (int): temporal stride between fast and slow pathways (usually 4)

    Returns:
        List[Tensor]: [slow_pathway, fast_pathway]
    """
    fast_pathway = frames # full frame sequence
    print(f"29. Fast Pathway Shape: {fast_pathway.shape}")
    slow_indices = torch.linspace(0, fast_pathway.shape[2] - 1, fast_pathway.shape[2] // 4).long().to(device)
    print(f"30. Line Spaced Slow Indices Using ({0, fast_pathway.shape[2] - 1, fast_pathway.shape[2] // 4})")
    slow_pathway = torch.index_select(fast_pathway, 2, slow_indices)  # T dimension is dim=2
    print(f"31. Slow Pathway Shape: {slow_pathway.shape}")

    print(f"32. Returning Pathways: [{slow_pathway.shape, fast_pathway.shape}]")
    return [slow_pathway, fast_pathway]

def run_inference(model, dataloader, idx_to_class, device):
    results = []
    with torch.no_grad():
        for INDEX, (videos, labels, paths) in enumerate(dataloader):
            print(f"-------- ITERATION {INDEX + 1} of 4 videos/batch ----------")
            # print("Checking Videos shape", videos.shape) #Checking Videos shape torch.Size([4, 32, 3, 224, 224])
            videos = videos.permute(0, 2, 1, 3, 4).to(device) #Checking Videos shape after  ze([4, 3, 32, 224, 224])
            # print("Checking Videos shape after permuting", videos.shape)
            inputs = pack_pathway_output(videos)
            outputs = model(inputs)
            print(f"33. Raw Outputs: {outputs}")
            preds = torch.argmax(outputs, dim=1)
            print(f"34. Predictions with torch.argmax: {preds}")
            for path, pred in zip(paths, preds):
                results.append((path, idx_to_class[pred.item()]))
            print("------------------------------------------------------------")

    print(f"35. Results: {results}")
    return results

def load_model(checkpoint_path, device, num_classes):

    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    print(f"25. Loading Model")
    
    # Replace the classification head
    in_features = model.blocks[-1].proj.in_features
    print(f"26. Infeatures, {in_features}")
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    print(f"27. Replaced the classification head with {in_features, num_classes}")
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model



# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the best.pt model")
    parser.add_argument("--data_dir", required=True, help="Path to the folder containing class folders with .mp4 videos")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    print(f"1. Printing Args: {args}")
    transform = Compose([
    Resize((256, 256)),
    CenterCrop(224),
    NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

    print(f"2. Applied Transform", transform)

    dataset = YourVideoDataset(args.data_dir, transform=transform, frames_per_clip=32)
    print(f"23. Dataset Received from Class: {dataset}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"24. Dataloaded: {dataloader}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.class_to_idx)
    model = load_model(args.checkpoint, device, num_classes)
    print(f"28. Model Loaded")

    results = run_inference(model, dataloader, dataset.idx_to_class, device)

    # Assuming you have a mapping from class indices to class names
    # idx_to_class = dataset.idx_to_class  # e.g. {0:'jump', 1:'run', 2:'walk'}
    # print("idx_to_class", idx_to_class)
    for path, pred_class_idx in results:
        print("36. pred-class_idx", pred_class_idx)
        # pred_class_name = idx_to_class[pred_class_idx]
        print(f"37. Video: {os.path.basename(path)} | Predicted Class: {pred_class_idx}")