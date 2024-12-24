import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
import argparse
import yaml
import os

# accept args to specify v1
parser = argparse.ArgumentParser(description='add weight directory')
parser.add_argument('--weights_dir', type=str, required=True, help='Specify the path to the safetensor file', default='weights/icon_detect')
args = parser.parse_args()

tensor_dict = load_file(os.path.join(args.weights_dir, "model.safetensors"))
model = DetectionModel(os.path.join(args.weights_dir, "model.yaml"))

model.load_state_dict(tensor_dict)
save_dict = {'model':model}

with open(os.path.join(args.weights_dir, "train_args.yaml"), 'r') as file:
    train_args = yaml.safe_load(file)
save_dict.update(train_args)
torch.save(save_dict, os.path.join(args.weights_dir, "best.pt"))

