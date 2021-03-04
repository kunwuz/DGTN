# DGTN: Dual-channel Graph Transition Network for Session-based Recommendation
This repository contains PyTorch Implementation of ICDMW 2020 paper: [*DGTN: Dual-channel Graph Transition Network for Session-based Recommendation*.](https://arxiv.org/abs/2009.10002)
Please check our paper for more details about our work if you are interested. 

## Usage
Following the steps below to run our codes:

###  1. Preprocess

The preprocess code is in `preprocess/`

###  2. Neighbors retrieval

Please run `neigh_retrieval/neighborhood_retrieval.py`

### 3. Run the model

Please run `main.py`

## Requirements
+ Python 3
+ PyTorch 1.1.0

## Citation
If you find this repo is useful for you, please kindly cite our paper.
```
@inproceedings{zheng2020dgtn,
    title={DGTN: Dual-channel Graph Transition Network for Session-based Recommendation},
    author={Zheng, Yujia and Liu, Siyi and Li, Zekun and Wu, Shu},
    booktitle={ICDMW},
    year={2020},
}
```
