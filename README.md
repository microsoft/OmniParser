# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<div style="text-align: center;">
  <img src="imgs/logo.png" alt="Logo">
</div>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/)] [[Models](https://huggingface.co/microsoft/OmniParser)] 

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [Upcoming] Both Interactive Region Detection Model and Icon functional description model will be release soon. 
- [2024/09] OmniParser achieves the best performance on [[Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)]! 

## Install 
```python
conda create -n "omni" python==3.12
pip install -r requirements.txt
```

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradion_demo.py
```


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
