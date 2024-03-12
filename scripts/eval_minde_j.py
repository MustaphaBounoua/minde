

from src.minde.minde_j import Minde_j
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import numpy as np
import argparse
import os
import jax
import json
import bmi
from .helper import *
jax.config.update('jax_platform_name', 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--task_0', type=int, default=0)
parser.add_argument('--task_n', type=int, default=-1)
parser.add_argument('--weighted', type=bool, default=False)
parser.add_argument('--debias_train', type=bool, default=True)
parser.add_argument('--debias_test',  type=bool, default=False)
parser.add_argument('--use_pretrain',  type=bool, default=False)
parser.add_argument('--preprocessing',  type=str, default="rescale")
parser.add_argument('--use_skip',  type=str, default=True)
parser.add_argument('--Train_Size',  type=int, default=100000 )
parser.add_argument('--Test_Size',  type=int, default=10000 )

parser.add_argument('--seed',  type=int, default=0)


NUM_epoch = [500, 750]

Train_Size = 50000
Test_Size = 10000

LR = [1e-3, 2e-3]

BATCH_SIZE = [128, 256]

BATCH_SIZE_TEST = 1000
NB_devices = 1
N_runs = 10
# SIGMAS = [0.5,1.0,1.5,2,3,5]
SIGMAS = [0.5, 1.0, 1.5, 2, 3, 5, 10]





def train_minde_or_load(args, task, train_l=None, test_l=None, pretrain=True, num_epoch=500):
    folder = "imp_{}_preprocess_{}_use_skip_{}/seed_{}/".format(
        args.debias_train, args.preprocessing, args.use_skip, args.seed)

    CHECKPOINT_DIR = BASE_Folder + folder+str(task.name)+"/"

    if task.dim_x <= 5:
        lr = LR[0]
    else:
        lr = LR[1]
    print(CHECKPOINT_DIR)
    if not pretrain:
        path = CHECKPOINT_DIR+"/mi_/" + "version_0/checkpoints/"

        tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                      name="mi_")

        mind = Minde_j(dim_x=task.dim_x, dim_y=task.dim_y, debias=args.debias_train, weighted=False, lr=lr,
                       use_skip=args.use_skip, use_ema=True, batch_size=BATCH_SIZE,
                       test_samples=get_samples(test_l, Test_Size), gt=task.mutual_information)
        mind = mind.to("cuda")
        mind.sde.device = mind.device
        get_trainer(tb_logger, CHECKPOINT_DIR, num_epoch).fit(
            model=mind, train_dataloaders=train_l, val_dataloaders=test_l)
    else:
        print("Using pretrained model")
        path = CHECKPOINT_DIR+"/mi_/" + ""
        mind = Minde_j.load_from_checkpoint(path, dim_x=task.dim_x, dim_y=task.dim_y, debias=args.debias_train, weighted=False,
                                            use_skip=args.use_skip, use_ema=True,
                                            test_samples=get_samples(test_l, Test_Size), gt=task.mutual_information)
        mind.use_ema = True
        mind = mind.to("cuda")
        mind.sde.device = mind.device

    return mind


def get_trainer(tb_logger, CHECKPOINT_DIR, num_epoch):
    return pl.Trainer(
        logger=tb_logger,
        accelerator='gpu', devices=NB_devices,
                    max_epochs=num_epoch,
                    default_root_dir=CHECKPOINT_DIR,
    )


def evaluate_task(args, sampler_task):

    train_l, test_l = get_data_loader(args, sampler_task)
    if sampler_task.dim_x <= 5:
        num_epoch = NUM_epoch[0]
    elif sampler_task.dim_x > 5:
        num_epoch = NUM_epoch[1]

    mind = train_minde_or_load(args, task=sampler_task,
                               train_l=train_l, test_l=test_l,
                               pretrain=args.use_pretrain, num_epoch=num_epoch)
    mind.to("cuda")
    mind.eval()
    samples = get_samples(test_l, n_sample=Test_Size)

    r = {}
    r["ema"] = {}
    r["non_ema"] = {}
    m_imp = []
    m = []

    m_s_imp = []
    m_s = []
    mind.use_ema = True
    r["ema"]["mi"] = {}
    r["ema"]["mi-imp"] = {}
    for sigma in SIGMAS:
        m_imp = []
        m = []
        for i in range(N_runs):
            m_imp.append(mind.mi_compute_non_square(
                samples, debias=True, sigma=sigma).cpu())
            m.append(mind.mi_compute_non_square(
                samples, debias=False, sigma=sigma).cpu())
        r["ema"]["mi"][sigma] = {"mean": float(np.mean(m)),
                                 "std":  float(np.std(m)),
                                 "max": float(np.max(m)),
                                 "min": float(np.min(m)),
                                 "s_run":  float(m[0])
                                 }
        r["ema"]["mi-imp"][sigma] = {"mean": float(np.mean(m_imp)),
                                     "std": float(np.std(m_imp)),
                                     "max": float(np.max(m_imp)),
                                     "min": float(np.min(m_imp)),
                                     "s_run":  float(m_imp[0])}

    for i in range(N_runs):
        m_s_imp.append(mind.mi_compute(samples, debias=True).cpu())
        m_s.append(mind.mi_compute(samples, debias=False).cpu())
    print(m)

    r["ema"]["mi-s"] = {"mean": float(np.mean(m_s)),
                        "std":  float(np.std(m_s)),
                        "max": float(np.max(m_s)),
                        "min": float(np.min(m_s)),
                        "s_run":  float(m_s[0])}

    r["ema"]["mi-imp-s"] = {"mean": float(np.mean(m_s_imp)),
                            "std": float(np.std(m_s_imp)),
                            "max": float(np.max(m_s_imp)),
                            "min": float(np.min(m_s_imp)),
                            "s_run":  float(m_s_imp[0])}

    mind.use_ema = False
    r["non_ema"]["mi"] = {}
    r["non_ema"]["mi-imp"] = {}
    for sigma in SIGMAS:
        m_imp = []
        m = []
        for i in range(N_runs):
            m_imp.append(mind.mi_compute_sigma(
                samples, debias=True, sigma=sigma).cpu())
            m.append(mind.mi_compute_sigma(
                samples, debias=False, sigma=sigma).cpu())
        r["non_ema"]["mi"][sigma] = {"mean": float(np.mean(m)),
                                     "std":  float(np.std(m)),
                                     "max": float(np.max(m)),
                                     "min": float(np.min(m)),
                                     "s_run":  float(m[0])
                                     }
        r["non_ema"]["mi-imp"][sigma] = {"mean": float(np.mean(m_imp)),
                                         "std": float(np.std(m_imp)),
                                         "max": float(np.max(m_imp)),
                                         "min": float(np.min(m_imp)),
                                         "s_run":  float(m_imp[0])}

    for i in range(N_runs):
        m_s_imp.append(mind.mi_compute(samples, debias=True).cpu())
        m_s.append(mind.mi_compute(samples, debias=False).cpu())
    print(m)

    r["non_ema"]["mi-s"] = {"mean": float(np.mean(m_s)),
                            "std":  float(np.std(m_s)),
                            "max": float(np.max(m_s)),
                            "min": float(np.min(m_s)),
                            "s_run":  float(m_s[0])}

    r["non_ema"]["mi-imp-s"] = {"mean": float(np.mean(m_s_imp)),
                                "std": float(np.std(m_s_imp)),
                                "max": float(np.max(m_s_imp)),
                                "min": float(np.min(m_s_imp)),
                                "s_run":  float(m_s_imp[0])}
    return r


if __name__ == "__main__":
    args = parser.parse_args()
    
    BASE_Folder = "runs/Minde_j_{}/".format(args.Train_Size)
    BASE_Folder_output = "results/Minde_j_{}/".format(args.Train_Size)

    tasks = list(bmi.benchmark.BENCHMARK_TASKS.keys())
    results_tasks = {}

    pl.seed_everything(args.seed)

    for task in tasks[args.task_0:args.task_n]:

        print(task)

        sampler_task = bmi.benchmark.BENCHMARK_TASKS[str(task)]

        print(f"Task {sampler_task.name} with dimensions {sampler_task.dim_x} and {sampler_task.dim_y} mi: {sampler_task.mutual_information} ")

        results = evaluate_task(args, sampler_task)

        results_tasks[sampler_task.name] = {
            "gt": sampler_task.mutual_information,
            "minde": results,
            "imp_train": args.debias_train,
            "preprocess": args.preprocessing,
            "weighted": args.weighted,
            "netarch_skip": args.use_skip
        }
        print(results_tasks)

    directory = BASE_Folder_output
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_path = directory+'/results_debais_train_{}_preprocess_{}_skip_{}_seed_{}/'.format(
        args.debias_train, args.preprocessing, args.use_skip, args.seed)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open("{}/results_-{}--{}.json".format(directory_path, args.task_0, args.task_n), 'w') as f:
        json.dump(results_tasks, f)
