import os
import subprocess
import time

# Define the list of files to download
files = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors"
]

# Define the target directory
weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
print(f"Storing weights at: {weights_dir}")

# Ensure the weights directory exists
os.makedirs(weights_dir, exist_ok=True)

# Loop through each file and download it using Hugging Face CLI
for file in files:
    print(f"Downloading {file}...")
    subprocess.run(["huggingface-cli", "download", "microsoft/OmniParser-v2.0", file, "--local-dir", weights_dir])
    time.sleep(1)  # Add a short delay to avoid rate-limiting issues

# Rename the directory "icon_caption" to "icon_caption_florence"
icon_caption_path = os.path.join(weights_dir, "icon_caption")
icon_caption_florence_path = os.path.join(weights_dir, "icon_caption_florence")

if os.path.exists(icon_caption_path):
    os.rename(icon_caption_path, icon_caption_florence_path)
    print(f"Renamed {icon_caption_path} to {icon_caption_florence_path}")
else:
    print(f"Directory {icon_caption_path} not found!")