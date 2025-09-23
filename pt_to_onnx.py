from pytorchvideo.models.hub import slowfast_r50
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Architecture + weights (for model saved using model.save) to onnx"""
# Load the model architecture+weights saved using torch.save(model)
# model = torch.load(
#     "/Models/3_15classes_archweights/arch.pt",
#     map_location="cpu"
# )
# model.eval()/

"""only weights (for model saved using model.load_state_dict)to onnx"""
# 1. Create base model
model = slowfast_r50()

# 2. Replace final classification head with one for 15 classes
# For SlowFast, the classifier is usually model.blocks[6]
model.blocks[6].proj = nn.Linear(model.blocks[6].proj.in_features, 15)

# 3. Load weights
state_dict = torch.load(r"/SlowfastTrainer/Models/Idle_E60_C16_Cam10718/architecture.pt", map_location="cpu")
model.load_state_dict(state_dict)

# 4. Set to evaluation mode
model.eval()

# Dummy input to trace the model
slow_input = torch.randn(4, 3, 8, 224, 224)   # [B, C, T, H, W]
fast_input = torch.randn(4, 3, 32, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    ([slow_input, fast_input],),                     # Model inputs
    r"/SlowfastTrainer/Models/Idle_E60_C16_Cam10718/Idle_E60_C16_Cam10718.onnx",                              # ✅ Path to save ONNX
    export_params=True,
    opset_version=11,
    input_names=["slow", "fast"],
    output_names=["output"],
    dynamic_axes={
        "slow": {0: "batch_size"},
        "fast": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ Model exported to slowfast.onnx")
