import bmi
import numpy as np
import json
import argparse
import os
from bmi.benchmark.tasks import transform_rescale, transform_uniformise, transform_gaussianise

# jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--task_0', type=int, default=0)
parser.add_argument('--task_n', type=int, default=1)
parser.add_argument('--weighted', type=bool, default=False)

parser.add_argument('--debias_train', type=bool, default=False)
parser.add_argument('--debias_test',  type=bool, default=False)

parser.add_argument('--use_pretrain',  type=bool, default=False)
parser.add_argument('--preprocessing',  type=str, default="None")
# Done in BMI

# parser.add_argument('--preprocessing',  type=str, default="gaussanize" )

parser.add_argument('--use_skip',  type=str, default=True)
SEED = 42

NUM_epoch = 500
# NUM_epoch = 1
# Train_Size = 256 * 10000
Train_Size = 100*1000
Test_Size = 10000

LR = 1e-2
BATCH_SIZE = 256
BATCH_SIZE_TEST = 1000
NB_devices = 1
N_runs = 1
# SIGMAS = [0.5,1.0,1.5,2,3,5]
SIGMAS = [1.0]


BASE_Folder_output = "results/output_baselines_/"


def get_data(args, task, seed=SEED):

    if args.preprocessing == "rescale":
        task = transform_rescale(task)
    elif args.preprocessing == "gaussanize":
        task = transform_gaussianise(task)
    elif args.preprocessing == "uniformize":
        task = transform_uniformise(task)

    size_train = Train_Size
    size_test = Test_Size

    X, Y = task.sample(size_test+size_train, seed=seed)

    x_train, y_train = X[:size_train, :], Y[:size_train, :]
    x_test, y_test = X[size_train:, :], Y[size_train:, :]

    return [x_train, y_train], [x_test, y_test]


def evaluate_task(args, sampler_task):
    if sampler_task.dim_x <= 10:
        dim = 32
        LR = 1e-3
        batch_size = 256
    elif sampler_task.dim_x <= 50:
        batch_size = 256
        LR = 1e-3
        dim = 128
    else:
        LR = 2e-3
        batch_size = 256
        dim = 256
    HIDDEN_layer = (dim, dim, 8)
    train_split = Train_Size/(Train_Size + Test_Size)
    steps = 20000 * 10
    test_every_n_steps = 250
    estimators_neural = {
        "MINE": bmi.estimators.MINEEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, max_n_steps=steps, batch_size=batch_size, train_test_split=train_split),
        "InfoNCE": bmi.estimators.InfoNCEEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, max_n_steps=steps, batch_size=batch_size, train_test_split=train_split),
        "D-V": bmi.estimators.DonskerVaradhanEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR,
                                                       max_n_steps=steps, batch_size=batch_size,
                                                       test_every_n_steps=1000,
                                                       train_test_split=train_split),
        "NWJ": bmi.estimators.NWJEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, max_n_steps=steps,
                                           test_every_n_steps=1000,
                                           batch_size=batch_size, train_test_split=train_split),
    }

    estimators_non_neural = {
        "KSG": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
        "LNN": bmi.estimators.KDEMutualInformationEstimator(),
        # "HIST (Julia)":bmi.estimators.HistogramEstimator(),
        #    "Transfer (Julia)" : bmi.estimators.HistogramEstimator()
        "CCA": bmi.estimators.CCAMutualInformationEstimator(),
    }
    r = {}
    results = {}
    for k in estimators_neural.keys():
        r[k] = []
        results[k] = {}

    for k in estimators_non_neural.keys():
        r[k] = []
        results[k] = {}
    SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for seed in SEEDS:
        estimators_neural = {
            "MINE": bmi.estimators.MINEEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, test_every_n_steps=test_every_n_steps,
                                                 max_n_steps=steps, seed=seed, batch_size=batch_size, train_test_split=train_split),
            "InfoNCE": bmi.estimators.InfoNCEEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, test_every_n_steps=test_every_n_steps,
                                                       max_n_steps=steps, seed=seed, batch_size=batch_size, train_test_split=train_split),
            "D-V": bmi.estimators.DonskerVaradhanEstimator(hidden_layers=HIDDEN_layer,
                                                           learning_rate=LR, max_n_steps=steps, seed=seed, test_every_n_steps=test_every_n_steps,
                                                           batch_size=batch_size, train_test_split=train_split),
            "NWJ": bmi.estimators.NWJEstimator(hidden_layers=HIDDEN_layer, learning_rate=LR, max_n_steps=steps,
                                               test_every_n_steps=test_every_n_steps,
                                               seed=seed, batch_size=batch_size, train_test_split=train_split),
        }
        estimators_non_neural = {
            "KSG": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
            "LNN": bmi.estimators.KDEMutualInformationEstimator(),
            # "HIST (Julia)":bmi.estimators.HistogramEstimator(),
            #    "Transfer (Julia)" : bmi.estimators.HistogramEstimator()
            "CCA": bmi.estimators.CCAMutualInformationEstimator(),
        }
        train_l, test_l = get_data(args, sampler_task, seed=seed)
        X = np.concatenate([train_l[0], test_l[0]])
        Y = np.concatenate([train_l[1], test_l[1]])

        for key in estimators_non_neural.keys():
            estimator = estimators_non_neural[key]
            mi = estimator.estimate(test_l[0], test_l[1])
            r[key].append(mi)
        for key in estimators_neural.keys():
            estimator = estimators_neural[key]
            mi = estimator.estimate(X, Y)
            r[key].append(mi)

    for k in estimators_neural.keys():
        m = r[k]
        results[k] = {"mean": float(np.mean(m)),
                      "std":  float(np.std(m)),
                      "max": float(np.max(m)),
                      "min": float(np.min(m)),
                      "s_run": float(m[0])}

    for k in estimators_non_neural.keys():
        m = r[k]
        results[k] = {"mean": float(np.mean(m)),
                      "std":  float(np.std(m)),
                      "max": float(np.max(m)),
                      "min": float(np.min(m)),
                      "s_run": float(m[0])
                      }

    return results


if __name__ == "__main__":
    args = parser.parse_args()

    tasks = list(bmi.benchmark.BENCHMARK_TASKS.keys())
    results_tasks = {}

    for task in tasks[args.task_0:args.task_n]:

        print(task)

        sampler_task = bmi.benchmark.BENCHMARK_TASKS[str(task)]

        print(f"Task {sampler_task.name} with dimensions {sampler_task.dim_x} and {sampler_task.dim_y} mi: {sampler_task.mutual_information} ")

        results = evaluate_task(args, sampler_task)

        results_tasks[sampler_task.name] = results
        print(results_tasks)

    directory = BASE_Folder_output
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_path = directory + \
        '/results2layers_preprocess_{}/'.format(args.preprocessing)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open("{}/results_-{}--{}.json".format(directory_path, args.task_0, args.task_n), 'w') as f:
        json.dump(results_tasks, f)
