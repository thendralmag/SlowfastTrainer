Overview
This Documentation is to understand train_slowfast.py, a fine tuning code for SlowFast video action recognition model using PyTorch. It supports fine-tuning on custom datasets organized into class folders, with logging of key metrics such as loss, accuracy, precision, recall, and F1 score.
The workflow includes:
Custom dataset loading with frame sampling.


Model preparation (loading pretrained SlowFast and replacing the classification head).


Training and validation loops with metric tracking.


Gradient checks and logging.


Saving the best-performing model.



File Structure
config.py → Contains hyperparameters (e.g., number of classes, paths, learning rate, optimizer).


YourVideoDataset → Custom dataset class for video frame sampling and preprocessing.


validate() → Validation function computing evaluation loss and metrics.


Training loop → Performs fine-tuning with gradient checks, logging, and model saving.



Dataset Handling
YourVideoDataset
This dataset class loads videos from a folder structure:
root_dir/
 ├── class_1/
 │    ├── video1.mp4
 │    ├── video2.mp4
 ├── class_2/
 │    ├── video3.mp4
 │    ├── video4.mp4

Each folder corresponds to a class.


Videos must be in .mp4 format.


Key Features:
Builds an index mapping classes → integer labels.


Uses Decord (VideoReader) for efficient frame extraction.


Handles short videos by padding frames.


Normalizes pixel values to [0,1].


Returns frames in shape (T, C, H, W).



Model
Base: Pretrained SlowFast-R50 from pytorchvideo.


Final classification head replaced with nn.Linear → num_classes.


Loaded on GPU (CUDA) if available, else CPU.



Optimizer
Default: Adam optimizer with learning rate from config.py.


Customizable for other optimizers.



Validation Function
validate(model, val_loader, criterion, device)
Evaluates model on validation set.


Computes:


Validation loss


Accuracy


Precision


Recall


F1-score


Uses macro averaging to treat all classes equally.



Training Pipeline
Sanity Check
Runs one forward + backward pass.


Ensures gradients flow through the final classification head.


Training Loop
For each epoch:
Training Phase


Videos processed into dual pathway (Slow + Fast).


Loss computed using criterion.


Backpropagation + optimizer step.


Tracks:


Training loss


Accuracy


Precision, Recall, F1


Gradient norms


Validation Phase


Uses validate() function.


Computes validation metrics.


Model Saving


Saves model weights (state_dict) and full model if validation loss improves.


Logging


Logs metrics to console and log_path file.


Includes:


Epoch number


Training & validation metrics


Gradient norm & learning rate


Time taken per epoch



Key Outputs
Logs: Training and validation metrics per epoch.


Best Model: Saved when validation loss improves.


model_save_path → Model weights


model_save_arch_path → Full model


Console Output: Training progress with metrics.



Metrics Computed
Training & Validation Loss


Accuracy


Precision (weighted/macro)


Recall (weighted/macro)


F1 Score (weighted/macro)


Gradient Norms



Time Tracking
Epoch-wise time.


Total training time in minutes.



Usage
Configure dataset paths and hyperparameters in config.py.


Ensure dataset folder structure is correct.


Run training script:

 python train_slowfast.py


Monitor logs for metrics and saved models.



Notes
Short videos are padded with repeated frames to ensure consistency.


Both slow pathway (subsampled frames) and fast pathway (all frames) are fed into SlowFast.


Best model checkpointing is based on lowest validation loss.


Logs provide full reproducibility of training process.
