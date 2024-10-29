import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
import os

tensor_dict = load_file(os.path.join("weights", "icon_detect", "model.safetensors"))

model = DetectionModel(os.path.join("weights", "icon_detect", "model.yaml"))
model.load_state_dict(tensor_dict)
torch.save({'model':model}, os.path.join("weights", "icon_detect", "model.pt"))
