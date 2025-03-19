# OmniParser: 純粋な視覚ベースのGUIエージェントのためのスクリーンパーシングツール

<p align="center">
  <img src="../imgs/logo.png" alt="ロゴ">
</p>

<p align="center">
[ <a href="../README.md">En</a> |
<a href="README_CN.md">中</a> |
<a href="README_FR.md">Fr</a> |
<b>日</b> ]
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[プロジェクトページ](https://microsoft.github.io/OmniParser/)] [[V2ブログ記事](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[モデルV2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[モデルV1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Spaceデモ](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser**は、ユーザーインターフェースのスクリーンショットを構造化され理解しやすい要素に解析する包括的な方法で、GPT-4Vがインターフェースの対応する領域に正確に基づいたアクションを生成する能力を大幅に向上させます。

## ニュース
- [2025/3] OmniToolにマルチエージェントのオーケストレーションを追加し、ユーザーインターフェースを改善してより良い体験を提供しています。
- [2025/2] OmniParser V2の[チェックポイント](https://huggingface.co/microsoft/OmniParser-v2.0)をリリースしました。[動画を見る](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] OmniToolを紹介します：OmniParser + 選択した視覚モデルでWindows 11 VMを制御します。OmniToolは以下の大規模言語モデルをサポートしています - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) または Anthropic Computer Use。[動画を見る](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2が登場します。新しいグラウンディングベンチマーク[Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main)で39.5%の新しい最先端の結果を達成しました（近日リリース予定）！詳細は[こちら](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md)をご覧ください。
- [2024/11] 更新版OmniParser V1.5をリリースしました。1) より細かい/小さなアイコン検出、2) 各画面要素が操作可能かどうかの予測が特徴です。デモ.ipynbに例があります。
- [2024/10] OmniParserはhuggingfaceモデルハブで#1トレンドモデルになりました（2024/10/29開始）。
- [2024/10] [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)でデモをチェックしてください！（OmniParser + Claude Computer Useにご期待ください）
- [2024/10] インタラクティブ領域検出モデルとアイコン機能説明モデルの両方がリリースされました！[Hugginfaceモデル](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParserが[Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)で最高のパフォーマンスを達成しました！

## インストール
まずリポジトリをクローンし、環境をインストールします：
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

V2の重みがweightsフォルダにダウンロードされていることを確認してください（caption weightsフォルダがicon_caption_florenceと呼ばれていることを確認してください）。ダウンロードされていない場合は、以下でダウンロードします：
```
   # モデルチェックポイントをローカルディレクトリOmniParser/weights/にダウンロードします
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [非推奨]
次に、モデルのckptsファイルを以下からダウンロードし、weights/の下に配置します。デフォルトのフォルダ構造は：weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2です。

v1の場合：
safetensorを.ptファイルに変換します。
```python
python weights/convert_safetensor_to_pt.py

v1.5の場合：
https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5から'model_v1_5.pt'をダウンロードし、新しいディレクトリweights/icon_detect_v1_5を作成し、そのフォルダ内に配置します。重みの変換は必要ありません。
``` -->

## 例：
demo.ipynbにいくつかの簡単な例をまとめました。

## Gradioデモ
Gradioデモを実行するには、以下のコマンドを実行します：
```python
python gradio_demo.py
```

## モデル重みのライセンス
huggingfaceモデルハブのモデルチェックポイントについては、icon_detectモデルは元のyoloモデルから継承されたライセンスであるためAGPLライセンスの下にあります。icon_caption_blip2とicon_caption_florenceはMITライセンスの下にあります。各モデルのフォルダ内のLICENSEファイルを参照してください：https://huggingface.co/microsoft/OmniParser。

## 📚 引用
技術レポートは[こちら](https://arxiv.org/abs/2408.00203)でご覧いただけます。
私たちの仕事が役に立った場合は、以下のように引用してください：
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