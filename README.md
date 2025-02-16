# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Models V1.5](https://huggingface.co/microsoft/OmniParser)] [[huggingface space (to be updated)](https://huggingface.co/spaces/microsoft/OmniParser)]

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface.

## News
- [2025/2] We release OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] We introduce OmniTool: Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 is coming. We achieve new state-of-the-art results 39.5% on the new grounding benchmark [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) with OmniParser v2 (will be released soon)! Read more details [here](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] We release an updated version, OmniParser V1.5 which features 1) more fine-grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb.
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024).
- [2024/10] Feel free to check out our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Huggingface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)!

## Install
### 1. Set Up Environment
Install the required Python environment:
```sh
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

### 2. Ensure PyTorch & Torchvision Are Installed Correctly
To avoid CUDA-related errors, ensure you install the correct version of PyTorch and Torchvision:

```sh
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
To verify the installation, run:
```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
Expected output:
```
2.1.0 12.1 True
```
If `torch.cuda.is_available()` returns `False`, your installation is incorrect.

### 3. Download Model Weights
Ensure you have the V2 weights downloaded into the `weights/` folder. Run the following commands:
```sh
rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights;
done
mv weights/icon_caption weights/icon_caption_florence
```

## Examples:
We put together a few simple examples in the `demo.ipynb`.

## Gradio Demo
To run the Gradio demo, execute:
```sh
python gradio_demo.py
```

## Model Weights License
For the model checkpoints on the Hugging Face model hub, please note that the **icon_detect model** is under an **AGPL license**, as it inherits from the original YOLO model. Meanwhile, **icon_caption_blip2** & **icon_caption_florence** are under an **MIT license**. Refer to the LICENSE file in the respective model folder: [Hugging Face Models](https://huggingface.co/microsoft/OmniParser).

## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing:
```bibtex
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

