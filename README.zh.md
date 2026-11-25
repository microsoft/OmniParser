# OmniParser：面向纯视觉 GUI 智能体的屏幕解析工具

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

📢 [[项目主页](https://microsoft.github.io/OmniParser/)] [[V2 博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[V2 模型](https://huggingface.co/microsoft/OmniParser-v2.0)] [[V1.5 模型](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space 演示](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** 是一种将用户界面（UI）屏幕截图解析为结构化且易于理解的元素的综合方法，它能够显著提升 GPT-4V 生成可在对应界面区域中精确定位（grounding）的操作的能力。

## 最新动态
- [2026/7] 新增了 YOLOv9-E 可交互区域检测器。其仅用于推理的权重可在 [Hugging Face PR #37](https://huggingface.co/microsoft/OmniParser-v2.0/discussions/37) 中获取。
- [2025/3] 支持轨迹（trajectory）的本地日志记录，方便您使用 OmniParser+OmniTool 为您所在领域的常用智能体构建训练数据流水线。[文档编写中]
- [2025/3] 正在逐步添加多智能体编排功能，并改进 OmniTool 的用户界面以提供更好的体验。
- [2025/2] 发布 OmniParser V2 [模型权重（checkpoints）](https://huggingface.co/microsoft/OmniParser-v2.0)。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] 推出 OmniTool：结合 OmniParser 与您自选的视觉模型来控制 Windows 11 虚拟机。OmniTool 开箱即支持以下大语言模型：OpenAI（4o/o1/o3-mini）、DeepSeek（R1）、Qwen（2.5VL）或 Anthropic Computer Use。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 即将推出。凭借 OmniParser v2（即将发布），我们在全新的视觉定位基准测试 [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) 上取得了 39.5% 的最新 SOTA 成绩！点击[此处](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md)阅读更多详情。
- [2024/11] 发布更新版本 OmniParser V1.5，其主要特性包括：1) 更细粒度/小图标的检测；2) 预测每个屏幕元素是否可交互。相关示例请参阅 `demo.ipynb`。
- [2024/10] OmniParser 登顶 Hugging Face 模型中心热门榜第 1 名（自 2024/10/29 起）。
- [2024/10] 欢迎体验我们在 [Hugging Face Space](https://huggingface.co/spaces/microsoft/OmniParser) 上的在线演示！（敬请期待 OmniParser + Claude Computer Use）
- [2024/10] 可交互区域检测模型（Interactive Region Detection Model）与图标功能描述模型（Icon functional description model）均已发布！[Hugging Face 模型](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser 在 [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/) 基准测试中取得了最佳性能！

## 安装
首先克隆本仓库，然后安装虚拟环境：
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

在 [Hugging Face PR #37](https://huggingface.co/microsoft/OmniParser-v2.0/discussions/37) 合并之前，请从该 PR 下载最新的 YOLOv9-E 检测器权重：
```
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect_v3/model.pt \
  --revision refs/pr/37 --local-dir weights
```

OmniParser 会优先使用该本地权重。待上述 PR 合并后，首次使用时会自动下载相同的权重。接着将图标描述（caption）模型权重下载到 `weights` 文件夹中：
```
   for f in icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
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

## 示例：
我们在 `demo.ipynb` 中整理了几个简单的使用示例。

## Gradio 演示
要运行 Gradio 演示，只需执行：
```python
python gradio_demo.py
```

## 模型权重许可证
`icon_detect_v3` 基于采用 MIT 许可证的 YOLOv9 实现。早期基于 Ultralytics 的图标检测器保留其原有的 AGPL 许可证。图标描述（caption）模型采用 MIT 许可证。

## 📚 引用
我们的技术报告可在[此处](https://arxiv.org/abs/2408.00203)查阅。
如果您觉得我们的工作对您有帮助，请考虑引用我们的论文：
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

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月16日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
