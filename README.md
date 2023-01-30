# TransGTR
This repo contains tpen-source code of TransGTR.

## Dependencies
- Python 3.8.5
- Pytorch 1.9.0

## Datasets
For double-blind reviewing, we only provide three public datasets, METR-LA, PEMS-BAY, and PEMSD7M. They can be downloaded from [DL-Traff](https://github.com/deepkashiwa20/DL-Traff-Graph). You can put the downloaded data in `TransGTR/METR-LA`, `TransGTR/PEMS-BAY` and `TransGTR/PEMSD7M`, respectively. 

## Steps to run TransGTR. 

### Step 0, Pre-process data. 
We provide data pre-processing scripts in `data_scripts/`. For example, if you want to 

### Step 1, Train a source feature network. 
If you want to use METR-LA or PEMS-BAY as source cities, you can obtain a pre-trained source feature network (TSFormer) from [STEP](https://github.com/zezhishao/STEP/tree/github/tsformer_ckpt). Otherwise, you should run the script 

`python3 train_tsformer.py --model TSFormer --data [YOUR DATA PATH]`

and set other parameters as you like. 

### Step 2, Train the node feature network $\theta_{nf}$ via knowledge distillation. 
To train the node feature network $\theta_{nf}$, you should run the script

`python3 train_distil.py --student_model DistilFormer --sdata [SOURCE DATA PATH] --tdata [TARGET DATA PATH] --teacher_model_path [SOURCE FEATURE NETWORK PATH] --data_number [DATA NUMBER]`

and set other parameters as you like. `DATA NUMBER` refers to the number of target data (in days). `DistilFormer` denotes the TSFormer enriched with day-in-week encodings. 

### Step 3, Train the forecasting model $\theta$ and the graph generator $\phi$ jointly. 
To train the forecasting model $\theta$ and the graph generator $\phi$ jointly, you should run the script 

`python3 train_forecast.py --sdata [SOURCE DATA PATH] --tdata [TARGET DATA PATH] --model DistilFormer --tsformer_path [NODE FEATURE NETWORK PATH] --data_number [DATA_NUMBER]`

and set other parameters as you like. 
