import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
from pathlib import Path

weights_path = Path("weights") / "icon_detect"

tensor_dict = load_file(weights_path / "model.safetensors")

model = DetectionModel(str(weights_path / "model.yaml"))
model.load_state_dict(tensor_dict)
torch.save({'model': model}, weights_path / "model.pt")
