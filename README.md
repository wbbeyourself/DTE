# Introduction
**D**etecting-**T**hen-**E**xplaining (**DTE**) aims to detect unanswerable and ambiguous spans in user question and give explainations to end-users by probing grounding knowledge from pretrained language models.

![Model Architecture](pictures/dte-model.png)




## Environment
config your local environment.
```bash
conda create -n dte python=3.7
conda activate dte
conda install pytorch==1.7.1    cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

set your own wandb key, get it from https://wandb.ai/home:
```bash
export WANDB_API_KEY=your_own_key_abcd
```

## Data Preparation
We put the data in `data` folder. You can download the data from [here](https://drive.google.com/file/d/1iucIjvQ7K1X5hxHdtmWhOWORmAVhUVEV/view?usp=sharing). Unzip the `data.zip` and put the subfolders in `data` folder.


## Try Train Model in Local GPU
Training scripts: `run_local.sh`.
Before your training, please figure out every command and arguments, and change user-related, path-related and gpu-device-related arguments to your own environment.
The training time is about 3 hours in 4 x 16G V100.

Usage:
```bash
chmod +x run_local.sh
./run_local.sh exp_name
```

