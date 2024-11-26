import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
import argparse
import yaml

# accept args to specify v1 or v1_5
parser = argparse.ArgumentParser(description='Specify version v1 or v1_5')
parser.add_argument('--version', choices=['v1', 'v1_5'], required=True, help='Specify the version: v1 or v1_5')
args = parser.parse_args()

if args.version == 'v1':
    tensor_dict = load_file("weights/icon_detect/model.safetensors")
    model = DetectionModel('weights/icon_detect/model.yaml')
    model.load_state_dict(tensor_dict)
    torch.save({'model':model}, 'weights/icon_detect/best.pt')
elif args.version == 'v1_5':
    print("Converting v1_5")
    tensor_dict = load_file("weights/icon_detect_v1_5/model.safetensors")
    model = DetectionModel('weights/icon_detect_v1_5/model.yaml')
    model.load_state_dict(tensor_dict)
    save_dict = {'model':model}

    with open("weights/icon_detect_v1_5/train_args.yaml", 'r') as file:
        train_args = yaml.safe_load(file)
    save_dict.update(train_args)
    torch.save(save_dict, 'weights/icon_detect_v1_5/best.pt')
