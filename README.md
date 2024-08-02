

[![arXiv](https://img.shields.io/badge/arXiv-2310.09031-b31b1b.svg)](https://arxiv.org/abs/2310.09031)
[![Venue](https://img.shields.io/badge/venue-ICLR_2024-darkblue)](https://iclr.cc/virtual/2024/poster/19605)

# MINDE: Mutual Information Neural Diffusion Estimation 


This repository contains the implementation for the paper [Mutual Information Neural Diffusion Estimation](https://arxiv.org/pdf/2402.05667) presented at ICLR 2024.


## Description

In this work we present a new method for the estimation of Mutual Information (MI) between random variables. Our approach is based on an original interpretation of the Girsanov theorem, which allows us to use score-based diffusion models to estimate the Kullback Leibler divergence between two densities as a difference between their score functions. As a by-product, our method also enables the estimation of the entropy of random variables. Armed with such building blocks, we present a general recipe to measure MI, which unfolds in two directions: one uses conditional diffusion process, whereas the other uses joint diffusion processes that allow simultaneous modelling of two random variables. Our results, which derive from a thorough experimental protocol over all the variants of our approach, indicate that our method is more accurate than the main alternatives from the literature, especially for challenging distributions. Furthermore, our methods pass MI self-consistency tests, including data processing and additivity under independence, which instead are a pain-point of existing methods.

## Installation

This repo was developed and tested under Python `3.9.12`. 


To install the `bmi` package : 

```bash
$ pip install benchmark-mi
```

To install the dependencies :

```bash
$ pip install -r requirements.txt
```



### Usage

Checkout  `quickstart.ipynb` for a quickstart on how to use MINDE.



First, default config can be loaded :
```python
args=get_default_config()
```

Choose one of the tasks in the benchmark  : 
```python
name_task = "1v1-normal-0.75"
task = bmi.benchmark.BENCHMARK_TASKS[name_task]
train_l,test_l = get_data_loader(args,task)
```
Groud truth information measures can be obtained:

```python 
from src.libs.minde import MINDE

minde = MINDE(args,var_list={"x":task.dim_x,"y":task.dim_y}, gt = task.mutual_information)
minde.fit(train_l,test_l)
minde.compute_mi()
```


### Running experiments

Running a particular experiment can be done using the scripts in `src/scripts/run_minde`. The experiments configurations are described in `src/scripts/config.py`


To run the experiments and reproduce results, an example shell script is provided in `src/scripts/job.sh`.


### Project Structure
```
minde/

├── src/                   # Source code for the project
│   ├── models/            # MLP with skip connection
│   ├── baseline           # MI neural estimators baseline 
│   ├── scripts            # Running MINDE
│   │    ├──config.py       # General configs
│   │    ├──helper.py       # helper functions
│   │    ├──run_minde.py    # script MINDE on several tasks
│   │    └──job.sh          # A shell job to run MINDE on several gpus on several tasks
│   └── libs               
│       ├──minde.py          # The main minde model class
│       ├──SDE.py          # The noising process which permits the learn the score functions
│       ├──info_measures.py # The set of functions to compute mutual information
│       ├──importance.py   # Required function to implement importance sampling scheme.
│       └──util.py         # General utility functions
├── quickstart.ipynb       # a jupyter notebook which explain how to use MINDE                           
├── requirements.txt       # List of dependencies
└── README.md              # This README file
```



## Cite our paper

```bibtex
@inproceedings{
minde2024,
title={{MINDE}: Mutual Information Neural Diffusion Estimation},
author={Giulio Franzese and Mustapha BOUNOUA and Pietro Michiardi},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=0kWd8SJq8d}
}
```
