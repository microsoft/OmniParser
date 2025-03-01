$files = @(
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors"
)

foreach ($f in $files) {
    huggingface-cli download microsoft/OmniParser-v2.0 $f --local-dir weights
}

Move-Item -Path "weights/icon_caption" -Destination "weights/icon_caption_florence"
