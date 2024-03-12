import torch
import numpy as np


def deconcat(z, mod_list, sizes):
    z_mods = {}
    idx = 0
    for i, mod in enumerate(mod_list):
        z_mods[mod] = z[:, idx:idx + sizes[i]]
        idx += sizes[i]
    return z_mods


def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat([z, encodings[key]], dim=-1)
    return z


def unsequeeze_dict(data):
    for key in data.keys():
        if data[key].ndim == 1:
            data[key] = data[key].view(data[key].size(0), 1)
    return data
