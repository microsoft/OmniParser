# OmniParser: 纯视觉GUI代理的屏幕解析工具

<p align="center">
  <img src="../imgs/logo.png" alt="Logo">
</p>

<p align="center">
[ <a href="../README.md">En</a> |
<b>中</b> |
<a href="README_FR.md">Fr</a> |
<a href="README_JA.md">日</a> ]
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[项目页面](https://microsoft.github.io/OmniParser/)] [[V2博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[V2模型](https://huggingface.co/microsoft/OmniParser-v2.0)] [[V1.5模型](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space演示](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** 是一种将用户界面截图解析为结构化且易于理解的元素的综合方法，显著增强了GPT-4V生成与界面相应区域准确对应的操作的能力。

## 新闻
- [2025/3] 我们正在逐步添加多代理编排功能，并改进OmniTool的用户界面，以提供更好的体验。
- [2025/2] 我们发布了OmniParser V2 [检查点](https://huggingface.co/microsoft/OmniParser-v2.0)。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] 我们推出了OmniTool：使用OmniParser + 您选择的视觉模型控制Windows 11虚拟机。OmniTool支持以下大型语言模型 - OpenAI (4o/o1/o3-mini)、DeepSeek (R1)、Qwen (2.5VL) 或 Anthropic Computer Use。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2即将发布。我们在新的基准测试[Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main)上取得了新的最先进结果，OmniParser v2达到了39.5%的准确率（即将发布）！阅读更多详情[此处](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md)。
- [2024/11] 我们发布了更新版本OmniParser V1.5，其特点包括：1) 更细粒度/小图标检测，2) 预测每个屏幕元素是否可交互。示例见demo.ipynb。
- [2024/10] OmniParser在huggingface模型中心成为#1热门模型（自2024年10月29日起）。
- [2024/10] 欢迎查看我们在[huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)上的演示！（敬请期待OmniParser + Claude Computer Use）
- [2024/10] 交互区域检测模型和图标功能描述模型均已发布！[Huggingface模型](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser在[Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)上取得了最佳表现！

## 安装
首先克隆仓库，然后安装环境：
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

确保V2权重已下载到weights文件夹中（确保caption权重文件夹名为icon_caption_florence）。如果未下载，请使用以下命令下载：
```
   # 将模型检查点下载到本地目录OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [已弃用]
然后下载模型ckpts文件：https://huggingface.co/microsoft/OmniParser，并将它们放在weights/下，默认文件夹结构为：weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2。

对于v1：
将safetensor转换为.pt文件。
```python
python weights/convert_safetensor_to_pt.py

对于v1.5：
从https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5下载'model_v1_5.pt'，新建一个目录：weights/icon_detect_v1_5，并将其放入该文件夹中。无需进行权重转换。
``` -->

## 示例：
我们在demo.ipynb中整理了一些简单的示例。

## Gradio演示
要运行gradio演示，只需运行：
```python
python gradio_demo.py
```

## 模型权重许可证
对于huggingface模型中心上的模型检查点，请注意icon_detect模型遵循AGPL许可证，因为它是从原始yolo模型继承的许可证。而icon_caption_blip2和icon_caption_florence遵循MIT许可证。请参考每个模型文件夹中的LICENSE文件：https://huggingface.co/microsoft/OmniParser。

## 📚 引用
我们的技术报告可以在[这里](https://arxiv.org/abs/2408.00203)找到。
如果您觉得我们的工作有用，请考虑引用我们的工作：
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