import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNIST_pairs(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x_1, x_2):
        self.x1 = x_1
        self.x2 = x_2

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return {"x1": self.x1[idx, :, :, :], "x2": self.x2[idx, :, :, :]}


def get_mnist_dataset(batch_size, train=None, drop_last=True):

    if train == None:
        # download and transform train dataset
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                  download=True,
                                                                  train=True,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                                      # normalize inputs
                                                                      transforms.Normalize(
                                                                          (0.1307,), (0.3081,))
                                                                  ])),
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=8, drop_last=drop_last, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                 download=True,
                                                                 train=False,
                                                                 transform=transforms.Compose([
                                                                     transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                                     # normalize inputs
                                                                     transforms.Normalize(
                                                                         (0.1307,), (0.3081,))
                                                                 ])),
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
        return train_loader, test_loader
    else:
        if train:
            return torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                              download=True,
                                                              train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                                  # normalize inputs
                                                                  transforms.Normalize(
                                                                      (0.1307,), (0.3081,))
                                                              ])),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        else:
            return torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                              download=True,
                                                              train=False,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                                  # normalize inputs
                                                                  transforms.Normalize(
                                                                      (0.1307,), (0.3081,))
                                                              ])),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
