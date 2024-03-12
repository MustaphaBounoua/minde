

[![arXiv](https://img.shields.io/badge/arXiv-2310.09031-b31b1b.svg)](https://arxiv.org/abs/2310.09031)
[![Venue](https://img.shields.io/badge/venue-ICLR_2024-darkblue)](https://iclr.cc/virtual/2024/poster/19605)

# Minde
Official implementation of MINDE: Mutual Information Neural Diffusion Estimation


## Requirements
List of packages (see also `requirements.txt`)

```
bmi==1.0.0
jax==0.4.25
numpy==1.26.4
pytorch_lightning==2.2.1
scikit_learn==1.4.1.post1
torch==2.2.1
torchvision==0.17.1
```


## Usage

MI estimation is conducted in `src\minde\minde_cond.py` and `src\minde\minde_j.py` with the functions `mi_compute_sigma()` and `mi_compute()` for both the variants (See Algorithm 2 and Algorithm 4 ).

## Demo 


`demo.ipynb` is a quick jupyter notebook to launch MINDE.

## Training


The training is conducted in `src\libs\train_step_cond.py` and `src\libs\SDE.train_step.py`( See Algorithm 1 and 3) 


Make sure the [requirements](#requirements) are satisfied in your environment. To run minde_c and minde_j and baselines, you can specify the number of tasks by choosing `task_0` and `task_n` which corresponds to the number of the task in in  [paper](https://arxiv.org/abs/2306.11078). 

```bash
python -m scripts.eval_minde_c --task_0 0 --task_n 2
python -m scripts.eval_minde_j --task_0 0 --task_n 2
python -m scripts.eval_baseline --task_0 0 --task_n 2

```

Please refer to `src/consistency_tests/` to run the consistency tests. Please note that it's first required to train the autoencoder by running `python -m src.consistency_tests.Autoencoder --rows 10`.

