# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Models V1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space Demo](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [2025/3] We support local logging of trajecotry so that you can use OmniParser+OmniTool to build training data pipeline for your favorate agent in your domain. [Documentation WIP]
- [2025/3] We are gradually adding multi agents orchstration and improving user interface in OmniTool for better experience.
- [2025/2] We release OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] We introduce OmniTool: Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 is coming. We achieve new state of the art results 39.5% on the new grounding benchmark [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) with OmniParser v2 (will be released soon)! Read more details [here](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] We release an updated version, OmniParser V1.5 which features 1) more fine grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb. 
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024). 
- [2024/10] Feel free to checkout our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)! 

## Install 
First clone the repo, and then install environment:
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:
```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## FastAPI Server (for integration)
An HTTP API wrapper is available for running OmniParser as a service with a response format compatible with the Minecraft Action Model loop.

Run the server (models auto-install on first run if missing):
```
python fastapi_server.py --device cuda --BOX_TRESHOLD 0.05 --host 0.0.0.0 --port 8510
python fastapi_server.py --som_model_path weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --host 0.0.0.0 --port 8510
```

Parse an image (multipart or JSON):
```
curl -F "image=@OmniParser/imgs/omni3.jpg" http://localhost:8510/parse
```

```
curl -X POST http://localhost:8510/parse \
  -H "Content-Type: application/json" \
  -d '{ "image_base64": "data:image/png;base64,..." }'
```

API endpoints
- `POST /parse`: parse image, returns `{ annotated_image_base64, annotation_list }`
- `GET /health`: readiness probe
- `GET /config`: current model configuration

Notes
- If the specified `som_model_path` or `caption_model_path` does not exist, the server downloads the appropriate files from `microsoft/OmniParser-v2.0` using Hugging Face Hub and updates the effective paths in `/config`.
- Requires network access for first-run downloads.

## Install with uv (CPU/GPU)
If you prefer `uv` over `conda`/`pip`, the project is configured with optional dependency groups for CPU and GPU installs via `pyproject.toml` extras: `cpu` and `gpu`.

Prereqs
- Python 3.10+ and `uv` installed (`pipx install uv` or see uv docs).

Create and activate a virtualenv
```
uv venv
source .venv/bin/activate
```

Base install (no heavy frameworks)
```
uv sync
```

CPU stack
```
uv pip install '.[cpu]'
```
Optionally lock then sync
```
uv lock --extra cpu
uv sync
```

GPU stack (CUDA)
- Choose your CUDA tag: `cu118`, `cu121`, or `cu122` (match your driver/runtime).
- Install the GPU extra with the proper indexes so PyTorch and Paddle use CUDA wheels:
```
uv pip install '.[gpu]' \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://www.paddlepaddle.org.cn/whl/cu121
```
Alternative two-step if you prefer to pin separately
```
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision
uv pip install -f https://www.paddlepaddle.org.cn/whl/cu121 paddlepaddle-gpu
uv pip install .
```

Verify GPU availability
```
uv run python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
uv run python - <<'PY'
import paddle
print('paddle cuda:', paddle.device.is_compiled_with_cuda())
print('device:', paddle.device.get_device())
PY
```

Notes
- `uv lock --extra gpu` records the dependency set, but wheel indexes come from the install command; keep the PyTorch `--extra-index-url` and Paddle `-f` in your deployment docs/scripts.
- Replace `cu121` in the URLs with your CUDA version as needed.
- Thanks to [PR #332](https://github.com/microsoft/OmniParser/pull/332) for clarifying which dependencies to pin in the GPU install instructions.

## Model Weights License
For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

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
