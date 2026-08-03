# OmniParser : Outil d'analyse d'écran pour agent GUI basé uniquement sur la vision

<p align="center">
  <img src="../imgs/logo.png" alt="Logo">
</p>

<p align="center">
[ <a href="../README.md">En</a> |
<a href="README_CN.md">中</a> |
<b>Fr</b> |
<a href="README_JA.md">日</a> ] 
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Page du projet](https://microsoft.github.io/OmniParser/)] [[Article de blog V2](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Modèles V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Modèles V1.5](https://huggingface.co/microsoft/OmniParser)] [[Démo HuggingFace Space](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** est une méthode complète pour analyser les captures d'écran d'interfaces utilisateur en éléments structurés et faciles à comprendre, ce qui améliore considérablement la capacité de GPT-4V à générer des actions pouvant être précisément ancrées dans les zones correspondantes de l'interface.

## Actualités
- [2025/3] Nous ajoutons progressivement une orchestration multi-agents et améliorons l'interface utilisateur dans OmniTool pour une meilleure expérience.
- [2025/2] Nous publions OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Regarder la vidéo](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] Nous présentons OmniTool : Contrôlez une VM Windows 11 avec OmniParser + votre modèle de vision choisi. OmniTool prend en charge nativement les modèles de langage suivants - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) ou Anthropic Computer Use. [Regarder la vidéo](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 arrive. Nous atteignons de nouveaux résultats de pointe avec 39.5% sur le nouveau benchmark d'ancrage [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) avec OmniParser v2 (sera bientôt publié) ! Lisez plus de détails [ici](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] Nous publions une version mise à jour, OmniParser V1.5, qui propose 1) une détection plus fine des petites icônes, 2) la prédiction de l'interactivité de chaque élément d'écran. Des exemples sont disponibles dans le fichier demo.ipynb.
- [2024/10] OmniParser était le modèle #1 tendance sur le hub de modèles HuggingFace (à partir du 29/10/2024).
- [2024/10] N'hésitez pas à consulter notre démo sur [HuggingFace Space](https://huggingface.co/spaces/microsoft/OmniParser) ! (restez à l'écoute pour OmniParser + Claude Computer Use)
- [2024/10] Les modèles de détection de région interactive et de description fonctionnelle des icônes sont publiés ! [Modèles HuggingFace](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser obtient les meilleures performances sur [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/) !

## Installation
Commencez par cloner le dépôt, puis installez l'environnement :
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Assurez-vous d'avoir les poids V2 téléchargés dans le dossier weights (assurez-vous que le dossier des poids de légende s'appelle icon_caption_florence). Sinon, téléchargez-les avec :
```
   # téléchargez les checkpoints du modèle dans le répertoire local OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [déprécié]
Ensuite, téléchargez les fichiers de checkpoints du modèle sur : https://huggingface.co/microsoft/OmniParser, et placez-les sous weights/, la structure de dossier par défaut est : weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2.

Pour v1 :
convertissez le safetensor en fichier .pt.
```python
python weights/convert_safetensor_to_pt.py

Pour v1.5 :
téléchargez 'model_v1_5.pt' depuis https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, créez un nouveau dossier : weights/icon_detect_v1_5, et placez-le dans ce dossier. Aucune conversion de poids n'est nécessaire.
``` -->

## Exemples :
Nous avons rassemblé quelques exemples simples dans le fichier demo.ipynb.

## Démo Gradio
Pour exécuter la démo Gradio, exécutez simplement :
```python
python gradio_demo.py
```

## Licence des poids du modèle
Pour les checkpoints de modèle sur le hub de modèles HuggingFace, veuillez noter que le modèle icon_detect est sous licence AGPL car il hérite de la licence du modèle yolo original. Et icon_caption_blip2 & icon_caption_florence sont sous licence MIT. Veuillez vous référer au fichier LICENSE dans le dossier de chaque modèle : https://huggingface.co/microsoft/OmniParser.

## 📚 Citation
Notre rapport technique peut être trouvé [ici](https://arxiv.org/abs/2408.00203).
Si vous trouvez notre travail utile, veuillez envisager de citer notre travail :
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