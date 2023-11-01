# TransGTR
This repo contains open-source code of TransGTR.

## Dependencies
- Python 3.8.5
- Pytorch 1.9.0

## Datasets
We only provide three public datasets, METR-LA, PEMS-BAY, and PEMSD7M. They can be downloaded from [DL-Traff](https://github.com/deepkashiwa20/DL-Traff-Graph). 

You can put the downloaded data in `data/METR-LA`, `data/PEMS-BAY` and `data/PEMSD7M`, respectively. 

You will need to run the following command to unzip the PEMS-BAY file as stated by DL-Traff. 

`unzip data/PEMS-BAY/pems-bay.zip`

## Contents of this repo
- `model.py` implements the base models, like the node feature network (`DistilTSFormer`), the graph generator (`DiscreteGraphLearningV2`), and the forecasting model. 
- `engine.py` implements the trainers of TransGTR. 
- `util.py` implements necessary utility functions, such as metrics and datasets. 
- `train_tsformer.py` implements the code for training source feature networks (in case pre-trained TSFormers are not available). 
- `train_distil.py` implements the code for training node feature networks $\theta_{nf}$ via knowledge distillation. 
- `train_forecast.py` implements the code for joint training of forecasting model $\theta$ and graph generator $\phi$. It includes both source training and fine-tuning. 

## Steps to run TransGTR. 

### Step 0, Pre-process data. 
We provide data pre-processing scripts in `data_scripts/`. For example, if you want to train the model with METR-LA as source and PEMSD7M as target, you should run the following data preprocessing scripts. 

`python3 data_scripts/generate_training_data_METR_LA.py --history_seq_len 2016 --future_seq_len 12`

`python3 data_scripts/generate_training_data_METR_LA.py --history_seq_len 12 --future_seq_len 12`

`python3 data_scripts/generate_training_PEMSD7M.py --history_seq_len 2016 --future_seq_len 12`

`python3 data_scripts/generate_training_PEMSD7M.py --history_seq_len 12 --future_seq_len 12`

where `--history_seq_len 2016` is used to train the node feature network, and `--history_seq_len 12` is used to train the forecasting model. 

### Step 1, Train a source feature network. 
If you want to use METR-LA or PEMS-BAY as source cities, you can obtain a pre-trained source feature network (TSFormer) from [STEP](https://github.com/zezhishao/STEP/tree/github/tsformer_ckpt). Otherwise, you should run the script 

`python3 train_tsformer.py --model TSFormer --data [YOUR DATA PATH]`

and set other parameters as you like. 

### Step 2, Train the node feature network $\theta_{nf}$ via knowledge distillation. 
To train the node feature network $\theta_{nf}$, you should run the script

`python3 train_distil.py --sdata [SOURCE DATA PATH] --tdata [TARGET DATA PATH] --teacher_model_path [SOURCE FEATURE NETWORK PATH] --data_number [DATA NUMBER]`

and set other parameters as you like. `DATA NUMBER` refers to the number of target data (in days). 

### Step 3, Train the forecasting model $\theta$ and the graph generator $\phi$ jointly. 
To train the forecasting model $\theta$ and the graph generator $\phi$ jointly, you should run the script 

`python3 train_forecast.py --sdata [SOURCE DATA PATH] --tdata [TARGET DATA PATH] --nfmodel DistilFormer --tsformer_path [NODE FEATURE NETWORK PATH] --data_number [DATA_NUMBER]`

and set other parameters as you like. 
