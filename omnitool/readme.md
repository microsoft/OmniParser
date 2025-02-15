<img src="../imgs/header_bar.png" alt="OmniTool Header" width="100%">

# OmniTool

Control a Windows 11 VM with OmniParser + your vision model of choice.

## Highlights:

1. **OmniParser V2** is 60% faster than V1 and now understands a wide variety of OS, app and inside app icons!
2. **OmniBox** uses 50% less disk space than other Windows VMs for agent testing, whilst providing the same computer use API
3. **OmniTool** supports out of the box the following vision models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use

## Overview

There are three components:

<table style="border-collapse: collapse; border: none;">
  <tr>
    <td style="border: none;"><img src="../imgs/omniparsericon.png" width="50"></td>
    <td style="border: none;"><strong>omniparserserver</strong></td>
    <td style="border: none;">FastAPI server running OmniParser V2.</td>
  </tr>
  <tr>
    <td style="border: none;"><img src="../imgs/omniboxicon.png" width="50"></td>
    <td style="border: none;"><strong>omnibox</strong></td>
    <td style="border: none;">A Windows 11 VM running in a Docker container.</td>
  </tr>
  <tr>
    <td style="border: none;"><img src="../imgs/gradioicon.png" width="50"></td>
    <td style="border: none;"><strong>gradio</strong></td>
    <td style="border: none;">UI to provide commands and watch reasoning + execution on OmniBox</td>
  </tr>
</table>

## Showcase Video
| OmniParser V2 | [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC) |
|--------------|------------------------------------------------------------------|
| OmniTool    | [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX) |


## Notes:

1. Though **OmniParser V2** can run on a CPU, we have separated this out if you want to run it fast on a GPU machine
2. The **OmniBox** Windows 11 VM docker is dependent on KVM so can only run quickly on Windows and Linux. This can run on a CPU machine (doesn't need GPU).
3. The Gradio UI can also run on a CPU machine. We suggest running **omnibox** and **gradio** on the same CPU machine and **omniparserserver** on a GPU server.

## Setup

1. **omniparserserver**:

   a. If you already have a conda environment for OmniParser, you can use that. Else follow the following steps to create one

   b. Ensure conda is installed with `conda --version` or install from the [Anaconda website](https://www.anaconda.com/download/success)

   c. Navigate to the root of the repo with `cd OmniParser`

   d. Create a conda python environment with `conda create -n "omni" python==3.12`

   e. Set the python environment to be used with `conda activate omni`

   f. Install the dependencies with `pip install -r requirements.txt`

   g. Continue from here if you already had the conda environment.

   h. Ensure you have the V2 weights downloaded in weights folder (**ensure caption weights folder is called icon_caption_florence**). If not download them with:
   ```
   rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence 
   for folder in icon_caption icon_detect; do huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; done
   mv weights/icon_caption weights/icon_caption_florence
   ```

   h. Navigate to the server directory with `cd OmniParser/omnitool/omniparserserver`

   i. Start the server with `python -m omniparserserver`

2. **omnibox**:

   a. Install Docker Desktop

   b. Visit [Microsoft Evaluation Center](https://info.microsoft.com/ww-landing-windows-11-enterprise.html), accept the Terms of Service, and download a **Windows 11 Enterprise Evaluation (90-day trial, English, United States)** ISO file [~6GB]. Rename the file to `custom.iso` and copy it to the directory `OmniParser/omnitool/omnibox/vm/win11iso`

   c. Navigate to vm management script directory with`cd OmniParser/omnitool/omnibox/scripts`

   d. Build the docker container [400MB] and install the ISO to a storage folder [20GB] with `./manage_vm.sh create`

   e. After creating the first time it will store a save of the VM state in `vm/win11storage`. You can then manage the VM with `./manage_vm.sh start` and `./manage_vm.sh stop`. To delete the VM, use `./manage_vm.sh delete` and delete the `OmniParser/omnitool/omnibox/vm/win11storage` directory.

3. **gradio**:

   a. Navigate to the gradio directory with `cd OmniParser/omnitool/gradio`

   b. Ensure you have activated the conda python environment with `conda activate omni`

   c. Start the server with `python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000`

   d. Open the URL in the terminal output, set your API Key and start playing with the AI agent!

## Risks and Mitigations
To align with the Microsoft AI principles and Responsible AI practices, we conduct risk mitigation by training the icon caption model with Responsible AI data, which helps the model avoid inferring sensitive attributes (e.g.race, religion etc.) of the individuals which happen to be in icon images as much as possible. At the same time, we encourage user to apply OmniParser only for screenshot that does not contain harmful/violent content. For the OmniTool, we conduct threat model analysis using Microsoft Threat Modeling Tool. We advise human to stay in the loop in order to minimize risk.


## Acknowledgment 
Kudos to the amazing resources that are invaluable in the development of our code: [Claude Computer Use](https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/README.md), [OS World](https://github.com/xlang-ai/OSWorld), [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena), and [computer_use_ootb](https://github.com/showlab/computer_use_ootb).
We are grateful for helpful suggestions and feedbacks provided by Francesco Bonacci, Jianwei Yang, Dillon DuPont, Yue Wu, Anh Nguyen. 
