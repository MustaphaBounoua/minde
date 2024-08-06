import bmi
import numpy as np
import json
from sklearn import preprocessing
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from src.baseline.doe import DoE
import torch
import torch.nn as nn


hidden = 2
dim = 64
layers = 2
lr = 1e-2
clip = 1
init = 0.0
carry = 0.99
alpha = 1.0
a = 'e'
SEED = 42
NUM_epoch = 500
# NUM_epoch = 1
# Train_Size = 256 * 10000
Train_Size = 50 * 1000
Test_Size = 10 * 1000

LR = 1e-2
BATCH_SIZE_TEST = 100
NB_devices = 1
N_runs = 1
# SIGMAS = [0.5,1.0,1.5,2,3,5]
SIGMAS = [1.0]
steps = 3000


BASE_Folder_output = "output_doe_50k/"


def control_weights(models):

    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and \
           init > 0.0:
            torch.nn.init.uniform_(m.weight, a=-init, b=init)

    for name in models:
        models[name].apply(init_weights)


def train_model(model, datal, lr=lr, clip=clip, nb_iter=steps):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    while step <= steps:
        for idx, batch in enumerate(datal):
            step += 1
            X = batch["x"].to("cuda")
            Y = batch["y"].to("cuda")
            XY = torch.cat([X.repeat_interleave(X.size(0), 0),
                            Y.repeat(Y.size(0), 1)], dim=1)

            loss = model(X, Y, XY)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
            if idx % 1000 == 0:
                print(loss)
    return model


def test_model(model, samples):
    model = model.eval()

    X = samples["x"].to("cuda")
    Y = samples["y"].to("cuda")
    XY = torch.cat([X.repeat_interleave(X.size(0), 0),
                    Y.repeat(X.size(0), 1)], dim=1)
    return - model(X, Y, XY).item()


# jax.config.update('jax_platform_name', 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--task_0', type=int, default=0)
parser.add_argument('--task_n', type=int, default=1)
parser.add_argument('--weighted', type=bool, default=False)

parser.add_argument('--debias_train', type=bool, default=False)
parser.add_argument('--debias_test',  type=bool, default=False)

parser.add_argument('--use_pretrain',  type=bool, default=False)
parser.add_argument('--preprocessing',  type=str, default="rescale")

parser.add_argument('--use_skip',  type=str, default=True)


class SynthetitcDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        self.x = torch.FloatTensor(np.array(data[0]))
        self.y = torch.FloatTensor(np.array(data[1]))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def get_data_loader(args, task, seed):

    size_train = Train_Size
    size_test = Test_Size

    X, Y = task.sample(size_test+size_train, seed=seed)

    if args.preprocessing == "rescale":
        X = preprocessing.StandardScaler(copy=True).fit_transform(X)
        Y = preprocessing.StandardScaler(copy=True).fit_transform(Y)

    x_train, y_train = X[:size_train,], Y[:size_train,]
    x_test, y_test = X[size_train:,], Y[size_train:,]

    data_train, data_test = [x_train, y_train], [x_test, y_test]

    train, test = SynthetitcDataset(data_train), SynthetitcDataset(data_test)

    batch_size_test = BATCH_SIZE_TEST

    train_loader = DataLoader(train, batch_size=128,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True
                              )

    test_loader = DataLoader(test, batch_size=100,
                             shuffle=False,
                             num_workers=8, drop_last=False,
                             pin_memory=True
                             )

    return train_loader, test_loader


def get_samples(test_loader, n_sample, device="cuda"):

    X = torch.Tensor()
    Y = torch.Tensor()

    for batch in test_loader:
        X = torch.cat([X, batch["x"]])
        Y = torch.cat([Y, batch["y"]])

    return {
        "x": X[:n_sample,],
        "y": Y[:n_sample,]
    }


def evaluate_task(args, sampler_task):
    if sampler_task.dim_x <= 10:
        hidden = 64
    elif sampler_task.dim_x <= 50:
        hidden = 128
    else:
        hidden = 256

    # hidden = 100
    layer = 2
    doe = DoE(sampler_task.dim_x, hidden, layer, 'gauss').to("cuda")
    doe_l = DoE(sampler_task.dim_x, hidden, layer, 'logistic').to("cuda")

    estimators_neural = {"doe_g": doe, "doe_lof": doe_l}
    control_weights(estimators_neural)
    r = {}
    results = {}
    for k in estimators_neural.keys():
        r[k] = []
        results[k] = {}

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        train_l, test_l = get_data_loader(args, sampler_task, seed)

        samples = get_samples(test_l, Test_Size)

        for key in estimators_neural.keys():
            estimators_neural[key] = train_model(
                estimators_neural[key], train_l, nb_iter=steps)
            mi = test_model(estimators_neural[key], samples)
            r[key].append(mi)

    for k in estimators_neural.keys():
        m = r[k]
        results[k] = {"mean": float(np.mean(m)),
                      "std":  float(np.std(m)),
                      "max": float(np.max(m)),
                      "min": float(np.min(m)),
                      "s_run": float(m[0]),
                      "gt": sampler_task.mutual_information}

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
        '/results_preprocess_{}/'.format(args.preprocessing)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open("{}/results_-{}--{}.json".format(directory_path, args.task_0, args.task_n), 'w') as f:
        json.dump(results_tasks, f)
