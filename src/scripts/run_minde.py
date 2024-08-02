
import pytorch_lightning as pl
import os
import jax
import json
import bmi
from .helper import *
from src.libs.minde import MINDE
from src.scripts.config import get_config
jax.config.update('jax_platform_name', 'cpu')

parser = get_config()


def evaluate_task(args, sampler_task):

    train_l, test_l = get_data_loader(args, sampler_task)
    if sampler_task.dim_x <= 5:
        args.max_epochs = 300
        args.lr = 1e-3
        args.bs = 128
    elif sampler_task.dim_x > 5:
        args.max_epochs = 500
        args.lr = 2e-3
        args.bs = 256
        
    args.benchmark =sampler_task.name 
    
    train_l,test_l = get_data_loader(args,sampler_task)
    minde = MINDE(args,var_list={"x":sampler_task.dim_x,"y":sampler_task.dim_y}, gt = sampler_task.mutual_information)
    minde.fit(train_l,test_l)

    
    minde.to("cuda")
    minde.eval()
    mi, mi_sigma = minde.compute_mi()
    return {"mi":mi,"mi_sigma":mi_sigma }



if __name__ == "__main__":
    args = parser.parse_args()
    
    
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
            "minde_estimate": results,
            "minde type": args.type,
            "importance_sampling": args.importance_sampling,
            "preprocess": args.preprocessing,
            "arch": args.arch
        }
        print(results_tasks)

    directory = args.results_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_path = directory+'/'.format(
        args.importance_sampling, args.preprocessing, args.arch, args.seed)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open("{}/results_-{}--{}.json".format(directory_path, args.task_0, args.task_n), 'w') as f:
        json.dump(results_tasks, f)
