#!/bin/bash

# download the model checkpoints to local directory OmniParser/weights/
rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence 
for folder in icon_caption icon_detect; do huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; done
mv weights/icon_caption weights/icon_caption_florence