# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/)] [[Models](https://huggingface.co/microsoft/OmniParser)] 

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)! 

### :rocket: Docker Quick Start

Prerequisites:
- CUDA-enabled GPU
- NVIDIA Container Toolkit installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```
# Build the image (requires CUDA)
sudo nvidia-docker build -t omniparser .

# Run the image
sudo docker run -d -p 7861:7861 --gpus all --name omniparser-container omniparser
```

## Install 
Install environment:
```python
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Download and convert the model ckpt files from https://huggingface.co/microsoft/OmniParser:
```python
python download.py
```

Or, download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2.
Finally, convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py
```

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## Deploy to AWS

To deploy OmniParser to EC2 on AWS via Github Actions:

1. Fork this repository and clone your fork to your local machine.
2. Follow the instructions at the top of [`deploy.py`](https://github.com/microsoft/OmniParser/blob/main/deploy.py).

## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
