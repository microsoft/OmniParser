# OmniParser+X Computer Use Demo

Control a Windows 11 VM with OmniParser+X (X = [GPT-4o/4o-mini, Claude, ...]).

## Overview

There are three components:

1. **windowshost**: A Windows 11 VM running in a Docker container
2. **omniparserserver**: FastAPI server running OmniParser
3. **gradio**: UI where you can provide commands and watch OmniParser+X reasoning and executing on the Windows 11 VM

Notes:
1. The Windows 11 VM docker is dependent on KVM so can only run quickly on Windows and Linux. This can run on a CPU machine (doesn't need GPU).
2. Though OmniParser can run on a CPU, we have separated this out if you want to run it fast on a GPU machine
3. The Gradio UI can also run on a CPU machine.

## Setup

1. **windowshost**:

   a. Install Docker Desktop
   
   b. Visit [Microsoft Evaluation Center](https://info.microsoft.com/ww-landing-windows-11-enterprise.html), accept the Terms of Service, and download a **Windows 11 Enterprise Evaluation (90-day trial, English, United States)** ISO file [~6GB]. Rename the file to `custom.iso` and copy it to the directory `OmniParser/computer_use_demo/windowshost/vm/win11iso`
   
   c. Navigate to vm management script directory with`cd OmniParser/computer_use_demo/windowshost/scripts`
   
   d. Build the docker container [400MB] and install the ISO to a storage folder [20GB] with `./manage_vm.sh create`
   
   e. After creating the first time it will store a save of the VM state in `vm/win11storage`. You can then manage the VM with `./manage_vm.sh start` and `./manage_vm.sh stop`. To delete the VM, use `./manage_vm.sh delete` and delete the `OmniParser/computer_use_demo/windowshost/vm/win11storage` directory.

2. **omniparserserver**:

   a. If you already have a conda environment for OmniParser, you can use that. Else follow the following steps to create one
   
   b. Ensure conda is installed with `conda --version` or install from the [Anaconda website](https://www.anaconda.com/download/success)
   
   c. Navigate to the root of the repo with `cd OmniParser`
   
   d. Create a conda python environment with `conda create -n "omni" python==3.12`
   
   e. Set the python environment to be used with `conda activate omni`
   
   f. Install the dependencies with `pip install -r requirements.txt`
   
   g. Continue from here if you already had the conda environment.
   
   h. Ensure you have the weights downloaded in weights folder. If not download them with `for folder in icon_caption_blip2 icon_caption_florence icon_detect icon_detect_v1_5; do huggingface-cli download microsoft/OmniParser --local-dir weights/"$folder" --repo-type model --include "$folder/*"; done`
   
   h. Navigate to the server directory with `cd OmniParser/computer_use_demo/omniparserserver`
   
   i. Start the server with `python -m omniparserserver`

3. **gradio**:

    a. Navigate to the gradio directory with `cd OmniParser/computer_use_demo/gradio`

    b. Ensure you have activated the conda python environment with `conda activate omni`

    c. Start the server with `python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000`

    d. Open the URL in the terminal output, set your API Key from OpenAI and start playing with the AI agent!
