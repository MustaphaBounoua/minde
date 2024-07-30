
import torch
import pytorch_lightning as pl
from src.model.score_net import UnetMLP
from src.libs.ema import EMA
from ..libs.SDE import VP_SDE
from ..libs.importance import get_normalizing_constant
from ..libs.util import concat_vect


class Minde_c(pl.LightningModule):

    def __init__(self, dim_x, dim_y, lr=1e-3, mod_list=["x", "y"],
                 debias=False, weighted=False, use_ema=False,
                 d=0.5, test_samples=None, gt=0.0, batch_size=64, plot_epoch=5
                 ):
        super(Minde_c, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.mod_list = mod_list
        self.gt = gt
        self.weighted = weighted

        dim = (dim_x + dim_y)
        if dim <= 10:
            hidden_dim = 64
        elif dim <= 50:
            hidden_dim = 128
        else:
            hidden_dim = 256

        time_dim = hidden_dim
        self.score = UnetMLP(dim=(dim_x + dim_y),
                             init_dim=hidden_dim,
                             dim_mults=[],
                             time_dim=time_dim, nb_mod=2,
                             out_dim=dim_x)

        self.d = d
        self.stat = None
        self.debias = debias
        self.lr = lr
        self.use_ema = use_ema
        self.plot_epoch = plot_epoch
        self.save_hyperparameters(
            "d", "debias", "lr", 
            "use_ema", "weighted", 
            "dim_x", "dim_y", 
            "gt", 
            "batch_size")

        self.test_samples = test_samples
        self.T = torch.nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=False)
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling=self.debias,
                          liklihood_weighting=False)

    def training_step(self, batch, batch_idx):

        self.train()

        loss = self.sde.train_step_cond(
            batch, self.score, d=self.d).mean()  # forward and compute loss

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

        loss = self.sde.train_step_cond(
            batch, self.score, d=self.d).mean()  # # forward and compute loss
        self.log("loss_test", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.score.parameters(), lr=self.lr, amsgrad=False)
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.plot_epoch == 0:
            mi = self.mi_compute(self.test_samples, debias=True)
           # mi_non_debias_square = self.mi_compute(self.test_samples,debias=False )
            mi_sigma_1 = self.mi_compute_sigma(
                self.test_samples, debias=True, sigma=1.0)
          #  mi_non_debias = self.mi_compute_sigma(self.test_samples,debias=False )
            print("Step :{} , GT:{}, MINDE_C :{} , MINDE_C_(sigma=1) :{}    ".format(
                self.global_step, self.gt, mi.item(), mi_sigma_1.item()))
            self.logger.experiment.add_scalars('Estimation mi',
                                               {'gt': self.gt,
                                                'mi': mi,
                                                "mi_sigma_1": mi_sigma_1,
                                                }, global_step=self.global_step)

    def mi_compute_sigma(self, data, debias=False, sigma=1.0, eps=1e-3):
        """MINDE_C : Compute MI using equation (18), difference outside.  

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

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, mean = self.sde.sample(t_n, data, mods_list)

        std_x = std["x"]
        mean_x = mean["x"]

        y_x = concat_vect({
            "x": Y["x"],
            "y": torch.zeros_like(Y["y"])}
        )

        y_xc = concat_vect({
            "x": Y["x"],
            "y": data["y"]}
        )

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)

        with torch.no_grad():
            if debias:
                a_x = - self.score_inference(y_x, t_n_x, None).detach()
                a_xy = - self.score_inference(y_xc, t_n_c, None).detach()

            else:
                a_x = - self.score_inference(y_x, t_n_x, std_x).detach()
                a_xy = - self.score_inference(y_xc, t_n_c, std_x).detach()

        N = x.size(1)
        M = x.size(0)

        # a_cond = concat_vect({"x":a_x["x"],"y":a_y["y"]})

        chi_t_x = mean_x ** 2 * sigma ** 2 + std_x**2
        ref_score_x = (Y["x"])/chi_t_x  # was *g

        if debias:
            # std = std["x"][:,0].reshape(t_.shape)
            const = get_normalizing_constant((1,), T=1-eps).to(x)

            e_x = -const * 0.5 * ((a_x + std_x * ref_score_x)**2).sum() / M

            e_xc = -const * 0.5 * ((a_xy + std_x * ref_score_x)**2).sum() / M

        else:
            g = g["x"].reshape(g["x"].size(0), 1)

            e_x = -0.5 * (g**2*(a_x + ref_score_x)**2).sum() / M

            e_xc = -0.5 * (g**2*(a_xy + ref_score_x)**2).sum() / M

        return e_x - e_xc

    def mi_compute(self, data, debias=False, eps=1e-3):
        """ MINDE_C : Compute MI using equation (18), difference inside.  

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

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, mean = self.sde.sample(t_n, data, mods_list)

        std_x = std["x"]

        y_x = concat_vect({
            "x": Y["x"],
            "y": torch.zeros_like(Y["y"])}
        )

        y_xc = concat_vect({
            "x": Y["x"],
            "y": data["y"]}
        )

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)

        with torch.no_grad():
            if debias:
                a_x = - self.score_inference(y_x, t_n_x, None).detach()
                a_xy = - self.score_inference(y_xc, t_n_c, None).detach()

            else:
                a_x = - self.score_inference(y_x, t_n_x, std_x).detach()
                a_xy = - self.score_inference(y_xc, t_n_c, std_x).detach()

        N = x.size(1)
        M = x.size(0)

        if debias:

            const = get_normalizing_constant((1,), T=1).to(x)

            est_score = const * 0.5 * ((a_x - a_xy)**2).sum() / M

        else:
            g = g["x"].reshape(g["x"].size(0), 1)

            est_score = 0.5 * (g**2*(a_x - a_xy)**2).sum() / M

        return est_score.detach()

    
    def fit(self,train_l,test_l):
        
        pl.Trainer(
            logger=TensorBoardLogger(save_dir=self.args.out_dir
                              name="mi_"+task.name),
            accelerator=self.args.accelerator,
            max_epochs=self.args.max_epochs,
            default_root_dir=self.args.out_dir,
        ).fit(model=minde, train_dataloaders=train_l,val_dataloaders=test_l  )