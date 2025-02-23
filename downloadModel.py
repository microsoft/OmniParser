import os
import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm

# Define the target directory for downloaded weights
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

# List of model files to download
files = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors"
]

# Hugging Face repository ID
repo_id = "microsoft/OmniParser-v2.0"

# Function to download a file with a progress bar
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Check if the file already exists and has the correct size
    if os.path.exists(save_path) and os.path.getsize(save_path) == total_size:
        print(f"✅ {os.path.basename(save_path)} already exists, skipping...")
        return

    # Download the file with a progress bar
    with open(save_path, "wb") as file, tqdm(
        desc=f"⬇ Downloading {os.path.basename(save_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))

# Start downloading files
for file in files:
    file_url = hf_hub_url(repo_id=repo_id, filename=file)
    local_path = os.path.join(weights_dir, file)

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download the file
    download_file(file_url, local_path)

# Rename the "icon_caption" directory to "icon_caption_florence" if it exists
icon_caption_path = os.path.join(weights_dir, "icon_caption")
icon_caption_florence_path = os.path.join(weights_dir, "icon_caption_florence")

if os.path.exists(icon_caption_path):
    os.rename(icon_caption_path, icon_caption_florence_path)
    print("🔄 Renamed 'icon_caption' to 'icon_caption_florence'.")

print("🎉 All downloads completed successfully!")
