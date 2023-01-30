# TransGTR
This repo contains tpen-source code of TransGTR.

## Dependencies
- Pytorch 1.9.0

## Datasets
For double-blind reviewing, we only provide three public datasets, METR-LA, PEMS-BAY, and PEMSD7M. They can be downloaded from [DL-Traff](https://github.com/deepkashiwa20/DL-Traff-Graph). 

## Steps to run TransGTR. 
1. If you want to use METR-LA or PEMS-BAY as source cities, you can obtain a pre-trained source feature network (TSFormer) from [STEP](https://github.com/zezhishao/STEP/tree/github/tsformer_ckpt). Otherwise, you should run the script 

`python3 train_tsformer.py --model TSFormer --data [YOUR DATA PATH]`

and set other parameters as you like. 
