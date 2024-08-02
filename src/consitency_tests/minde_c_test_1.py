

import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.mlp import UnetMLP_simple
from src.libs.util import EMA
from src.libs.SDE import VP_SDE, concat_vect, deconcat
from ..libs.importance import get_normalizing_constant
from .Autoencoder import AE, MnistDecoder, MnistEncoder, log_modalities
from src.data.mnist_pair import get_mnist_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
import json
import argparse

T0 = 1


parser = argparse.ArgumentParser()
parser.add_argument('--rows',  type=int, default=0)
parser.add_argument('--seed',  type=int, default=0)


class Minde_mnist_c(pl.LightningModule):

    def __init__(self, dim_x, dim_y, lr=1e-3, var_list=["x", "y"], use_skip=True,
                 debias=False, weighted=False, use_ema=False,
                 d=0.5, test_samples=None, gt=0.0, aes=None,
                 rows=28
                 ):
        super(Minde_mnist_c, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.var_list = var_list
        self.gt = gt
        self.weighted = weighted

        hidden_dim = 256
        time_dim = hidden_dim
        self.score = UnetMLP_simple(dim=(dim_x + dim_y), init_dim=hidden_dim,
                                 dim_mults=(1, 1), time_dim=time_dim, nb_mod=2, out_dim=dim_x)

    
        
        self.stat = None
        self.debias = debias
        self.lr = lr
        self.use_ema = use_ema
        self.rows = rows

        self.save_hyperparameters(
            "d", "debias", "lr", "use_ema", "weighted", "dim_x", "dim_y", "gt", "rows")

        self.aes = aes

        self.test_samples = self.get_mod_cropped_dataset(test_samples)
        self.T = torch.nn.Parameter(
            torch.FloatTensor([T0]), requires_grad=False)
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling=self.debias,
                          liklihood_weighting=False)

    def get_mod_cropped_dataset(self, loader):
        X = torch.Tensor().to(self.device)
        Y = torch.Tensor().to(self.device)
        nb_lines = self.rows
        for batch in loader:
            x = batch[0].to(self.device)
            y = x.clone().to(self.device)

            y[:, :, nb_lines:, :] = 0.0

            X = torch.cat([X, x])
            Y = torch.cat([Y, y])

        return {
            "x": X,
            "y": Y
        }

    def get_mod_cropped(self, x):

        y = x.clone()
        nb_lines = self.rows
        y[:, :, nb_lines:, :] = 0.0
        return {
            "x": x,
            "y": y
        }

    def encode(self, x):
        with torch.no_grad():
            latent_z = {}
            for mod in self.var_list:
                latent_z[mod] = self.aes[mod].encode(x[mod]).detach()
            return latent_z

    def decode(self, z):
        with torch.no_grad():
            output = {}
            for mod in self.var_list:
                output[mod] = self.aes[mod].decode(z[mod]).detach()
            return output

    def standerdize(self, z):
        if self.stat:
            for mod in self.var_list:
                z[mod] = (z[mod] - self.stat[mod]["mean"]) / \
                    self.stat[mod]["std"]
        return z

    def destanderdize(self, z):
        if self.stat:
            for mod in self.var_list:
                z[mod] = (z[mod] * self.stat[mod]["std"]) + \
                    self.stat[mod]["mean"]
        return z

    def training_step(self, batch, batch_idx):

        self.train()
        data = self.get_mod_cropped(batch[0])
        z = self.encode(data)

        if self.global_step == 0:
            self.stat = {}
            for mod in self.var_list:
                self.stat[mod] = {
                    "mean": z[mod].mean(dim=0),
                    "std": z[mod].std(dim=0),
                }

            std = self.stat[mod]["std"].clone()
            self.stat[mod]["std"][std == 0] = 1.0

            print(self.stat)
            with open(os.path.join(self.logger.log_dir, "stat.pickle"), "wb") as f:
                pickle.dump(self.stat, f)
        z = self.standerdize(z)

        loss = self.sde.train_step_cond(z, self.score, d=self.d).mean()
        # forward and compute loss
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
        batch = self.get_mod_cropped(batch[0])
        z = self.encode(batch)
        z = self.standerdize(z)
        loss = self.sde.train_step_cond(
            z, self.score, d=self.d).mean()  # # forward and compute loss
        self.log("loss_test", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.score.parameters(), lr=self.lr, amsgrad=False)
        return optimizer

    def log_samples(self):
        x_c = torch.rand((8, self.dim_x + self.dim_y)).to(self.device)
        x_c = self.standerdize(
            deconcat(x_c, self.var_list, sizes=[self.dim_x, self.dim_y]))
        x_c = concat_vect(x_c)

        if self.use_ema:
            self.model_ema.module.eval()
            z = self.sde.sample_euler_c(x_c, self.model_ema.module)
        else:
            self.score.eval()
            z = self.sde.sample_euler_c(x_c, self.score)

        z_samp = self.destanderdize(
            deconcat(z, var_list=["x", "y"], sizes=[self.dim_x, self.dim_y]))
        output = self.decode(z_samp)
        log_modalities(self.logger, output, [
                       "x", "y"], self.current_epoch, prefix="sampling/", nb_samples=8)

        test_samp_small = {
            "x": self.test_samples["x"][:8].to(self.device),
            "y": self.test_samples["y"][:8].to(self.device)
        }
        log_modalities(self.logger, test_samp_small, [
                       "x", "y"], self.current_epoch, prefix="real/", nb_samples=8)

        z_c = self.encode(test_samp_small)

        decode_ae = self.decode(z_c)

        log_modalities(self.logger, decode_ae, [
                       "x", "y"], self.current_epoch, prefix="real_ae/", nb_samples=8)

        z_c = self.standerdize(z_c)
        z_c = concat_vect(z_c)
        masks = [
            self.get_mask([self.dim_x, self.dim_y], subset=[
                          0], shape=z.shape).to(self.device),
            self.get_mask([self.dim_x, self.dim_y], subset=[
                          1], shape=z.shape).to(self.device)
        ]

       # x1 = z_c * masks[0] + torch.randn_like(z).to(z) * (1.0 - masks[0])
        x2 = z_c * masks[1] + torch.randn_like(z).to(z) * (1.0 - masks[1])

        # cond_in_0=  self.destanderdize(deconcat(x1,var_list=["x","y"],sizes= [self.dim_x,self.dim_y]) )
        cond_in_1 = self.destanderdize(
            deconcat(x2, var_list=["x", "y"], sizes=[self.dim_x, self.dim_y]))
      #  cond_in_out_1 =self.decode(cond_in_0)
        cond_in_out_2 = self.decode(cond_in_1)
       # log_modalities(self.logger, cond_in_out_1, ["x","y"], self.current_epoch ,prefix="cond_0_in/" ,nb_samples=8)
        log_modalities(self.logger, cond_in_out_2, [
                       "x", "y"], self.current_epoch, prefix="cond_1_in/", nb_samples=8)

        if self.use_ema:
           # output_cond_0 = self.sde.modality_inpainting(score_net=self.model_ema.module,x = x1 , mask = masks[0],  subset=[0])
            output_cond_1 = self.sde.modality_inpainting_c(
                score_net=self.model_ema.module, x=x2, mask=masks[1], subset=[1])
        else:
          #  output_cond_0 = self.sde.modality_inpainting(score_net=self.score,x = x1 , mask = masks[0],  subset=[0])
            output_cond_1 = self.sde.modality_inpainting_c(
                score_net=self.score, x=x2, mask=masks[1], subset=[1])

       # cond_samp_0=  self.destanderdize(deconcat(output_cond_0,var_list=["x","y"],sizes= [self.dim_x,self.dim_y]) )
        cond_samp_1 = self.destanderdize(
            deconcat(output_cond_1, var_list=["x", "y"], sizes=[self.dim_x, self.dim_y]))

     #   output_cond_0_im = self.decode(cond_samp_0)
        output_cond_1_im = self.decode(cond_samp_1)

      #  log_modalities(self.logger, output_cond_0_im, ["x","y"], self.current_epoch ,prefix="cond_0/" ,nb_samples=8)
        log_modalities(self.logger, output_cond_1_im, [
                       "x", "y"], self.current_epoch, prefix="cond_1/", nb_samples=8)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.sde.device = self.device
        if self.current_epoch % 25 == 0:
            self.log_samples()

        if self.current_epoch % 5 == 0:
            self.test_samples["x"] = self.test_samples["x"].to(self.device)
            self.test_samples["y"] = self.test_samples["y"].to(self.device)
            # self.test_samples =self.test_samples.to(self.device)

            z = self.encode(self.test_samples)

            z = self.standerdize(z)

            mi_debias_square = self.mi_compute(z, debias=True)
            mi_non_debias_square = self.mi_compute(z, debias=False)
            mi_debias = self.mi_compute_non_square(z, debias=True)
            mi_non_debias = self.mi_compute_non_square(z, debias=False)
            r = {'gt': self.gt,  'mi_imp': float(mi_debias.cpu().numpy()),
                 'mi': float(mi_non_debias.cpu().numpy()),
                 "mi_square_imp": float(mi_debias_square.cpu().numpy()),
                 "mi_square": float(mi_non_debias_square.cpu().numpy()), }

            self.logger.experiment.add_scalars('Estimation mi',
                                               r, global_step=self.global_step)

            if self.current_epoch % 100 == 0:
                with open(os.path.join(self.logger.log_dir, "results_{}.json".format(self.current_epoch)), 'w') as fp:
                    json.dump(r, fp)

    def get_mask(self, modalities_list_dim, subset, shape):
        mask = torch.zeros(shape)
        idx = 0
        for index_mod, dim in enumerate(modalities_list_dim):
            if index_mod in subset:
                mask[:, idx:idx + dim] = 1.0
            idx = idx + dim
        return mask

    def mi_compute_non_square(self, data, debias=False, sigma=1.0, eps=1e-5):

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

    def mi_compute(self, data, debias=False, eps=1e-5):

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

        # a_cond = concat_vect({"x":a_x["x"],"y":a_y["y"]})

        if debias:
            # std = std["x"][:,0].reshape(t_.shape)
            const = get_normalizing_constant((1,), T=1).to(x)

            est_score = const * 0.5 * ((a_x - a_xy)**2).sum() / M

        else:
            g = g["x"].reshape(g["x"].size(0), 1)

            est_score = 0.5 * (g**2*(a_x - a_xy)**2).sum() / M

        return est_score.detach()


def get_stat(file_name):
    t = pickle.load(open(file_name, "rb"))
    for key in t.keys():
        if key != 'cat':
            t[key]['mean'] = t[key]['mean'].to("cuda")
            t[key]['std'] = t[key]['std'].to("cuda")
    return t


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    pl.seed_everything(args.seed)
    Batch_size = 64
    NUM_epoch = 600
    rows = args.rows
    dim = 16
    LR = 1e-3
    paths = [
        "trained_models/ae_mnist_rows/crop_28/version_0/checkpoints/epoch=99-step=93700.ckpt", ##full image
        "trained_models/ae_mnist_rows/crop_{}/version_0/checkpoints/epoch=99-step=93700.ckpt".format(
            rows)
    ]
    
    train_l, test_l = get_mnist_dataset(batch_size=Batch_size)

    train_samp = next(iter(train_l))[0][:8,]
    test_samp = next(iter(test_l))[0][:8,]

    ae_1 = AE.load_from_checkpoint(paths[0]).eval()
    ae_2 = AE.load_from_checkpoint(paths[1]).eval()

    mld = Minde_mnist_c(var_list=["x", "y"],
                      dim_x=dim, dim_y=dim, lr=LR, aes=nn.ModuleDict({
                          "x": ae_1, "y": ae_2
                      }), rows=rows,
                      test_samples=test_l, gt=rows/28, use_ema=True,
                      debias=True,
                      weighted=False, d=0.5)
    CHECKPOINT_DIR = "runs/trained_models/mld_c_mi_row/"+str(args.seed)+"/"

    tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                  name="crop_"+str(rows)
                                  )

 
    trainer = pl.Trainer(
        logger=tb_logger,
        check_val_every_n_epoch=25,
        accelerator='gpu',
        devices=1,
      
        max_epochs=NUM_epoch,
        default_root_dir=CHECKPOINT_DIR)

    trainer.fit(model=mld, train_dataloaders=train_l,
                val_dataloaders=test_l,
               
                )
