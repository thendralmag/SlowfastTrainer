# SlowFast Video Action Recognition - Fine-tuning

## Overview

This documentation covers `train_slowfast.py`, a fine-tuning implementation for the **SlowFast video action recognition model** using PyTorch. The script enables fine-tuning on custom datasets organized into class folders, with comprehensive logging of key metrics including loss, accuracy, precision, recall, and F1 score.

### Workflow
1. Custom dataset loading with frame sampling
2. Model preparation (loading pretrained SlowFast and replacing the classification head)
3. Training and validation loops with metric tracking
4. Gradient checks and logging
5. Saving the best-performing model

## File Structure

- **config.py** → Contains hyperparameters (e.g., number of classes, paths, learning rate, optimizer)
- **YourVideoDataset** → Custom dataset class for video frame sampling and preprocessing
- **validate()** → Validation function computing evaluation loss and metrics
- **Training loop** → Performs fine-tuning with gradient checks, logging, and model saving

## Dataset Handling

### YourVideoDataset

This dataset class loads videos from a folder structure:

```
root_dir/
 ├── class_1/
 │    ├── video1.mp4
 │    ├── video2.mp4
 ├── class_2/
 │    ├── video3.mp4
 │    ├── video4.mp4
```

- Each folder corresponds to a class
- Videos must be in **.mp4** format

### Key Features:
- Builds an index mapping classes → integer labels
- Uses **Decord** (VideoReader) for efficient frame extraction
- Handles short videos by padding frames
- Normalizes pixel values to [0,1]
- Returns frames in shape (T, C, H, W)

## Model

- **Base**: Pretrained SlowFast-R50 from pytorchvideo
- Final classification head replaced with `nn.Linear` → `num_classes`
- Loaded on **GPU (CUDA)** if available, else CPU

## Optimizer

- **Default**: Adam optimizer with learning rate from config.py
- Customizable for other optimizers

## Validation Function

### `validate(model, val_loader, criterion, device)`

Evaluates model on validation set and computes:
- Validation loss
- Accuracy
- Precision
- Recall
- F1-score

Uses **macro averaging** to treat all classes equally.

## Training Pipeline

### Sanity Check
- Runs one forward + backward pass
- Ensures gradients flow through the final classification head

### Training Loop

For each epoch:

1. **Training Phase**
   - Videos processed into **dual pathway (Slow + Fast)**
   - Loss computed using criterion
   - Backpropagation + optimizer step
   - Tracks:
     - Training loss
     - Accuracy
     - Precision, Recall, F1
     - Gradient norms

2. **Validation Phase**
   - Uses `validate()` function
   - Computes validation metrics

3. **Model Saving**
   - Saves model weights (`state_dict`) and full model if validation loss improves

4. **Logging**
   - Logs metrics to console and `log_path` file
   - Includes:
     - Epoch number
     - Training & validation metrics
     - Gradient norm & learning rate
     - Time taken per epoch

## Key Outputs

1. **Logs**: Training and validation metrics per epoch
2. **Best Model**: Saved when validation loss improves
   - `model_save_path` → Model weights
   - `model_save_arch_path` → Full model
3. **Console Output**: Training progress with metrics

## Metrics Computed

- **Training & Validation Loss**
- **Accuracy**
- **Precision (weighted/macro)**
- **Recall (weighted/macro)**
- **F1 Score (weighted/macro)**
- **Gradient Norms**

## Time Tracking

- Epoch-wise time
- Total training time in minutes

## Usage

1. Configure dataset paths and hyperparameters in **config.py**
2. Ensure dataset folder structure is correct
3. Run training script:
   ```bash
   python train_slowfast.py
   ```
4. Monitor logs for metrics and saved models

## Important Notes

- Short videos are padded with repeated frames to ensure consistency
- Both **slow pathway (subsampled frames)** and **fast pathway (all frames)** are fed into SlowFast
- Best model checkpointing is based on **lowest validation loss**
- Logs provide full reproducibility of training process

## Requirements

- PyTorch
- pytorchvideo
- Decord
- scikit-learn (for metrics)
- CUDA (optional, for GPU acceleration)

## Directory Structure Example

```
project/
├── config.py
├── train_slowfast.py
├── dataset/
│   ├── class_1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── class_2/
│       ├── video3.mp4
│       └── video4.mp4
├── models/
│   ├── best_model_weights.pth
│   └── best_model_architecture.pth
└── logs/
    └── training_log.txt
```
