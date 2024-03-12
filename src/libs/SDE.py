
import torch
import random

from .util import *
from .importance import *


class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 liklihood_weighting=False,
                 nb_mod=2
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.liklihood_weighting = liklihood_weighting
        self.device = "cuda"
        self.nb_mod = nb_mod
        self.t_epsilon = 1e-3

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        # return mean std of p(x(t))
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

        log_mean_coeff = log_mean_coeff.to(self.device)

        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean * torch.ones_like(x).to(self.device), std.view(-1, 1) * torch.ones_like(x).to(self.device)

    def sample(self, t, data, mods_list):

        nb_mods = len(mods_list)
        self.device = t.device

        x_t_m = {}
        std_m = {}
        mean_m = {}
        z_m = {}
        g_m = {}

        for i, mod in enumerate(mods_list):
            x_mod = data[mod]

            z = torch.randn_like(x_mod).to(self.device)
            f, g = self.sde(t[:, i])

            mean_i, std_i = self.marg_prob(
                t[:, i].view(x_mod.shape[0], 1), x_mod)

            std_m[mod] = std_i
            mean_m[mod] = mean_i
            z_m[mod] = z
            g_m[mod] = g
            x_t_m[mod] = mean_i * x_mod + std_i * z

        return x_t_m, z_m, std_m, g_m, mean_m

    def train_step(self, data, score_net, eps=1e-5, d=0.5):
        # data= unsequeeze_dict(data)
        x = concat_vect(data)

        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]

        nb_mods = len(mods_list)

        if self.importance_sampling:
            t = self.sample_debiasing_t(shape=(x.shape[0], 1)) .to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x.shape[0], 1)) + eps).to(self.device)

        t_n = t.expand((x.shape[0], nb_mods))

        learn_cond = (torch.bernoulli(torch.tensor([d])) == 1.0)
        mask = [1, 1]
        if learn_cond:
            subsets = [[0, 1], [1, 0]]
            i = random.randint(0, len(mods_list)-1)
            mask = subsets[i]
            mask_time = torch.tensor(mask).to(self.device).expand(t_n.size())
            t_n = t_n * mask_time

        x_t_m, z_m, std_m, g_m, mean_m = self.sample(
            t=t_n, data=data, mods_list=mods_list)

        score = - score_net(concat_vect(x_t_m), t=t_n, std=None)

        weight = 1.0
        if learn_cond:

            score_m = deconcat(score, mods_list, mods_sizes)
            for idx, i in enumerate(mask):

                if i == 0:
                    # all the benchmark has two equal size mods
                    dim_clean = score_m[mods_list[idx]].size(1)
                    z_m.pop(mods_list[idx])
                    score_m.pop(mods_list[idx])

                else:
                    dim_diff = score_m[mods_list[idx]].size(1)
            weight += dim_clean/dim_diff
            score = concat_vect(score_m)

        loss = weight * \
            torch.square(score + concat_vect(z_m)).sum(1, keepdim=False)

        return loss

    def train_step_cond(self, data, score_net, eps=1e-3, d=0.5):
        # data= unsequeeze_dict(data)
        x = concat_vect(data)

        mods_list = list(data.keys())

        nb_mods = len(mods_list)

        if self.importance_sampling:
            t = (self.sample_debiasing_t(
                shape=(x.shape[0], 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x.shape[0], 1)) + eps).to(self.device)

        t_n = t.expand((x.shape[0], nb_mods))

        mask = [1, 0]

        x_t_m, z_m, std_m, g_m, mean_m = self.sample(
            t=t_n, data=data, mods_list=mods_list)

        learn_cond = (torch.bernoulli(torch.tensor([d])) == 1.0)
        if learn_cond:
            mask = [1, 0]
            mask_time = torch.tensor(mask).to(self.device).expand(t_n.size())
            t_n = t_n * mask_time + 1.0 * (1 - mask_time)
            # print(learn_cond)
            # print(t_n)
            x_t = concat_vect({"x": x_t_m["x"],
                               "y": data["y"]})
        else:
            mask = [1, 0]
            mask_time = torch.tensor(mask).to(self.device).expand(t_n.size())
            t_n = t_n * mask_time + 0.0 * (1 - mask_time)

            x_t = concat_vect({"x": x_t_m["x"],
                               "y": torch.zeros_like(data["y"])})

            # print(learn_cond)
            # print(t_n)

        score = - score_net(x_t, t=t_n, std=None)
        weight = 1.0
        loss = weight * torch.square(score + z_m["x"]).sum(1, keepdim=False)
        return loss

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)
