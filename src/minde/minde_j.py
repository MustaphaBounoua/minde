
import torch
import pytorch_lightning as pl
from src.model.score_net import UnetMLP
from src.libs.ema import EMA
from ..libs.SDE import VP_SDE
from ..libs.importance import get_normalizing_constant
from ..libs.util import concat_vect, deconcat


T0 = 1


class Minde_j(pl.LightningModule):

    def __init__(self, dim_x, dim_y, lr=1e-3, mod_list=["x", "y"], use_skip=True,
                 debias=False, weighted=False, use_ema=False,
                 d=0.5, test_samples=None, gt=0.0, batch_size=64,
                 ):
        super(Minde_j, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.mod_list = mod_list
        self.gt = gt
        self.weighted = weighted

        if use_skip == True:
            dim = (dim_x + dim_y)
            if dim <= 10:
                hidden_dim = 64
            elif dim <= 50:
                hidden_dim = 128
            else:
                hidden_dim = 256

            time_dim = hidden_dim
            self.score = UnetMLP(dim=(
                dim_x + dim_y), init_dim=hidden_dim, dim_mults=[], time_dim=time_dim, nb_mod=2)

        self.d = d
        self.stat = None
        self.debias = debias
        self.lr = lr
        self.use_ema = use_ema

        self.save_hyperparameters(
            "d", "debias", "lr", "use_ema", "weighted", "dim_x", "dim_y", "gt", "batch_size")

        self.test_samples = test_samples
        self.T = torch.nn.Parameter(
            torch.FloatTensor([T0]), requires_grad=False)
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling=self.debias,
                          liklihood_weighting=False)

    def training_step(self, batch, batch_idx):

        self.train()

        # forward and compute loss
        loss = self.sde.train_step(batch, self.score, d=self.d).mean()

        self.log("loss", loss)

        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    def score_inference(self, x, t, std):
        with torch.no_grad():
            self.eval()
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x, t, std)
            else:
                return self.score(x, t, std)

    def validation_step(self, batch, batch_idx):
        self.eval()

        # # forward and compute loss
        loss = self.sde.train_step(batch, self.score, d=self.d).mean()
        self.log("loss_test", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.score.parameters(), lr=self.lr, amsgrad=False)
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % 20 == 0:
            mi_debias_square = self.mi_compute(self.test_samples, debias=True)
            mi_non_debias_square = self.mi_compute(
                self.test_samples, debias=False)
            mi_debias = self.mi_compute_sigma(self.test_samples, debias=True)
            mi_non_debias = self.mi_compute_sigma(
                self.test_samples, debias=False)

            self.logger.experiment.add_scalars('Estimation mi',
                                               {'gt': self.gt,
                                                'mi_imp': mi_debias,
                                                'mi': mi_non_debias,
                                                "mi_square_imp": mi_debias_square,
                                                "mi_square": mi_non_debias_square,
                                                }, global_step=self.global_step)

    def mi_compute_sigma(self, data, debias=False, sigma=1.0, eps=1e-5):
        """MINDE_J : Compute MI using equation (20), difference outside.  

        Args:
            data (Dictionnary): Data samples
            debias (bool, optional): Using importance sampling. Defaults to False.
            eps (float, optional): Using for uniforme time sampling. Defaults to 1e-3.
            sigma (float, optional)

        Returns:
            _type_: MI estimation
        """
        self.sde.device = self.device
        self.score.eval()

        x, y = data["x"], data["y"]

        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]
        nb_mods = len(mods_list)

        if debias:
            t_ = self.sde.sample_debiasing_t(
                [x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device)
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]
                            ).to(self.device) * (self.T - eps) + eps
            # t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) *self.T+1e-3

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, mean = self.sde.sample(t_n, data, mods_list)

        y_xy = concat_vect(Y)
        std_xy = concat_vect(std)
        mean_xy = concat_vect(mean)

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        mask_time_y = torch.tensor([0, 1]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x
        t_n_y = t_n * mask_time_y

        y_x = concat_vect({
            "x": Y["x"],
            "y": y
        })

        y_y = concat_vect({
            "x": x,
            "y": Y["y"]
        })

        with torch.no_grad():
            if debias:
                a_xy = - self.score_inference(y_xy, t_n, None).detach()
                a_x = - self.score_inference(y_x, t_n_x, None).detach()
                a_y = - self.score_inference(y_y, t_n_y, None).detach()

            else:
                a_xy = - self.score_inference(y_xy, t_n, std_xy).detach()
                a_x = - self.score_inference(y_x, t_n_x, std_xy).detach()
                a_y = - self.score_inference(y_y, t_n_y, std_xy).detach()

        N = x.size(1)
        M = x.size(0)

        a_x = deconcat(a_x, mods_list, mods_sizes)["x"]
        a_y = deconcat(a_y, mods_list, mods_sizes)["y"]

        # a_cond = concat_vect({"x":a_x["x"],"y":a_y["y"]})

        chi_t_x = mean["x"] ** 2 * sigma ** 2 + std["x"]**2
        ref_score_x = (Y["x"])/chi_t_x  # was *g

        chi_t_y = mean["y"] ** 2 * sigma ** 2 + std["y"]**2
        ref_score_y = (Y["y"])/chi_t_y  # was *g

        chi_t_xy = mean_xy ** 2 * sigma ** 2 + std_xy**2
        ref_score_xy = (y_xy)/chi_t_xy  # was *g

        if debias:
            # std = std["x"][:,0].reshape(t_.shape)
            const = get_normalizing_constant((1,), T=1).to(x)

            e_x = -const * 0.5 * ((a_x + std["x"] * ref_score_x)**2).sum() / M

            e_y = -const * 0.5 * ((a_y + std["y"]*ref_score_y)**2).sum() / M

            e_xy = -const * 0.5 * ((a_xy + std_xy*ref_score_xy)**2).sum() / M
        else:
            g = g["x"].reshape(t_.shape)

            e_x = -0.5 * (g**2*(a_x + ref_score_x)**2).sum() / M

            e_y = -0.5 * (g**2*(a_y + ref_score_y)**2).sum() / M

            e_xy = - 0.5 * (g**2 * (a_xy + ref_score_xy)**2).sum() / M

        return e_xy - e_x - e_y

    def mi_compute(self, data, debias=False, eps=1e-3):
        """MINDE_C : Compute MI using equation (21), difference inside.  

        Args:
            data (Dictionnary): Data samples
            debias (bool, optional): Using importance sampling. Defaults to False.
            eps (float, optional): Using for uniforme time sampling. Defaults to 1e-3.

        Returns:
            _type_: MI estimation
        """
        self.sde.device = self.device
        self.score.eval()
        x, y = data["x"], data["y"]

        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]
        nb_mods = len(mods_list)

        if debias:
            t_ = self.sde.sample_debiasing_t(
                [x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device)
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]
                            ).to(self.device) * (self.T - eps) + eps
            # t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) *self.T+1e-3

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, _ = self.sde.sample(t_n, data, mods_list)

        y_xy = concat_vect(Y)
        std_xy = concat_vect(std)

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        mask_time_y = torch.tensor([0, 1]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x
        t_n_y = t_n * mask_time_y

        y_x = concat_vect({
            "x": Y["x"],
            "y": y
        })

        y_y = concat_vect({
            "x": x,
            "y": Y["y"]
        })

        with torch.no_grad():
            if debias:
                a_xy = - self.score_inference(y_xy, t_n, None).detach()
                a_x = - self.score_inference(y_x, t_n_x, None).detach()
                a_y = - self.score_inference(y_y, t_n_y, None).detach()
            else:
                a_xy = - self.score_inference(y_xy, t_n, std_xy).detach()
                a_x = - self.score_inference(y_x, t_n_x, std_xy).detach()
                a_y = - self.score_inference(y_y, t_n_y, std_xy).detach()

        N = x.size(1)
        M = x.size(0)

        a_x = deconcat(a_x, mods_list, mods_sizes)
        a_y = deconcat(a_y, mods_list, mods_sizes)

        a_cond = concat_vect({"x": a_x["x"], "y": a_y["y"]})

        if debias:

            const = get_normalizing_constant((1,)).to(x)
            est_score = const * 0.5 * ((a_xy - a_cond)**2).sum() / M
        else:
            g = g["x"].reshape(t_.shape)
            est_score = 0.5 * (g**2 * (a_xy - a_cond)**2).sum() / M * self.T

        return est_score.detach()
