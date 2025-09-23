import os
import random
import shutil
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

# Mapping class folder names to labels
CLASS_DICT = {
    'Bicep_Curls': 0,
    'DB_Chest_Press': 1,
    'DB_Incline_Chest_Press': 2,
    'DB_Lunges': 3,
    'DB_Reverse_Flys': 4,
    'Front_Raises': 5,
    'Hammer_Curls': 6,
    'KB_Goblet_Squats': 7,
    'KB_Overhead_Press': 8,
    'KB_Swings': 9,
    'KB_Goodmorning': 10,
    'Lateral_Raise': 11,
    'Seated_DB_Shoulder_Press': 12,
    'Singlearm_DB_Rows': 13,
    'Upright_Rows': 14,
    'Idle': 15
}

def split_dataset(data_root, output_root, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    data_root = Path(data_root)
    output_root = Path(output_root)
    print("data_rooot", data_root)
    print("output root", output_root)

    video_paths = []
    for class_name in CLASS_DICT:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist.")
            continue
        for file in os.listdir(class_dir):
            print(file)
            print(class_dir)
            print(class_dir/file)
            if file.endswith(".mp4"):
                video_paths.append((class_dir / file, class_name))

    # print(f"Video Path list \n, {video_paths}")

    # Shuffle and split
    train_val_paths, test_paths = train_test_split(video_paths, test_size=1 - (train_ratio + val_ratio), random_state=seed, shuffle=True)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed, shuffle=True)

    # print("train_path", train_paths)
    # print("val_path", val_paths)
    # print("test_path", test_paths)

    sets = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths
    }

    for split_name, samples in sets.items():
        split_dir = output_root / split_name
        os.makedirs(split_dir, exist_ok=True)
        csv_path = output_root / f"{split_name}.csv"

        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for src_path, class_name in samples:
                dst_dir = split_dir / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / src_path.name
                shutil.copy2(src_path, dst_path)

                relative_path = f"{class_name}/{src_path.name}"
                class_idx = CLASS_DICT[class_name]
                writer.writerow([relative_path, class_idx])

        print(f"{split_name} set: {len(samples)} videos | CSV saved to {csv_path}")

if __name__ == "__main__":
    DATASET_FOLDER = "/home/smartan/Desktop/SlowfastTrainer/Exercise_Recognition_Dataset_Cam107-18"  # Replace this with your actual dataset root
    OUTPUT_FOLDER = "/home/smartan/Desktop/SlowfastTrainer/DatasetIdle_16Classes_Cam107-18"  # Where train/val/test folders and CSVs will go
    split_dataset(DATASET_FOLDER, OUTPUT_FOLDER)



########################################################################################################################################################################################

"""

🔹 Input

Dataset root folder (DATASET_FOLDER)

Must contain subfolders, one for each action class.

Each class subfolder must contain .mp4 videos of that action.

Example structure:

Exercise_Recognition_Dataset_Cam107-18/
├── Bicep_Curls/
│   ├── vid1.mp4
│   ├── vid2.mp4
├── DB_Chest_Press/
│   ├── vid3.mp4
├── Idle/
│   ├── vid4.mp4
...


Class dictionary (CLASS_DICT)

Maps class folder names → integer labels (0, 1, 2, …).

Example:

'Bicep_Curls': 0,
'DB_Chest_Press': 1,
...
'Idle': 15

🔹 Processing Steps

Collect all videos from the dataset root, along with their class labels.

Split the dataset into:

Training (70%)

Validation (15%)

Test (15%)
(Ratios adjustable in split_dataset() arguments.)

Copy videos into structured output folders:

DatasetIdle_16Classes_Cam107-18/
├── train/
│   ├── Bicep_Curls/
│   ├── DB_Chest_Press/
│   ├── ...
├── val/
│   ├── ...
├── test/
│   ├── ...


Generate CSV annotation files (train.csv, val.csv, test.csv)

Each row contains:

<relative_path_to_video> <class_index>


Example row:

Bicep_Curls/vid1.mp4 0

🔹 Output

After running the script, the OUTPUT_FOLDER will look like:

DatasetIdle_16Classes_Cam107-18/
├── train/
│   ├── Bicep_Curls/
│   │   ├── vid1.mp4
│   ├── DB_Chest_Press/
│   │   ├── vid2.mp4
│   ├── ...
├── val/
│   ├── ...
├── test/
│   ├── ...
├── train.csv
├── val.csv
├── test.csv

🔹 Why This is Suitable for SlowFast?

SlowFast training expects:

Videos organized in per-class subfolders OR

A CSV file with paths and labels.

This script provides both:

Structured train/val/test directories.

Annotation CSVs in the format:

path label


Fully compatible with PySlowFast’s video_loader and custom dataset loaders.

✅ With this setup, you can directly point your SlowFast training config to the train.csv, val.csv, and test.csv files.


"""