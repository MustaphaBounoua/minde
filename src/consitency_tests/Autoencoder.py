import sys
root = "/home/*****/work/mi/"
sys.path.append(root)
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from src.data.mnist_pair import get_mnist_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rows',  type=int, default=0 )

LATENT_DIM =64
INPUT_DATA_DIM = 784
dataSize = torch.Size([1, 28, 28])
num_hidden_layers = 1






class MnistEncoder(nn.Module):
    def __init__(self, input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM, deterministic = False):
        super(MnistEncoder, self).__init__()
        self.deterministic = deterministic
        self.hidden_dim = 400;

        modules = []
        modules.append(nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True)))
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.relu = nn.ReLU();
  
        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim, bias=True)
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim, bias=True)


    def forward(self, x):
        h = x.view(*x.size()[:-3], -1);
        h = self.enc(h);
        h = h.view(h.size(0), -1);

        latent_space_mu = self.hidden_mu(h);
        latent_space_logvar = self.hidden_logvar(h);
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        if self.deterministic == True:
            return latent_space_mu
        else:
            return latent_space_mu, latent_space_logvar;



class MnistDecoder(nn.Module):
    def __init__(self, input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM):
        super(MnistDecoder, self).__init__();
      
        self.hidden_dim = 400;
        modules = []
 
        modules.append(nn.Sequential(nn.Linear(latent_dim, self.hidden_dim), nn.ReLU(True)))

        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.relu = nn.ReLU();
        self.sigmoid = nn.Sigmoid();

    def forward(self, z):
        x_hat = self.dec(z);
        x_hat = self.fc3(x_hat);
        x_hat = self.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *dataSize)
        return x_hat







import torch
import pytorch_lightning as pl

MODEL_STR = "AE"
        
def log_modalities(logger,output,mod_list,epoch,nb_samples=4, prefix="sampling/"):
    
    for mod in mod_list:
        data_mod = output[mod].cpu()[:nb_samples,]
        ready_to_plot = torchvision.utils.make_grid( data_mod.view(data_mod.size(0), 1, 28, 28),8  )
        logger.experiment.add_image(prefix + mod, ready_to_plot, global_step=epoch)
        
class AE(pl.LightningModule):

    def __init__(self, mod_name,test_loader = None, enc= None,dec = None,latent_dim=64,rows = 28,
                 regularization = None , alpha = 0.0 , lr =0.001, decay =0.0 ,train_loader = None):

        super(AE, self).__init__()
        self.lr = lr
        self.decay = decay
        self.modality = mod_name
        self.latent_dim =  latent_dim
        self.encoder = enc
        self.decoder = dec
        self.regularization = regularization
        self.test_loader = test_loader
        self.alpha = alpha
        self.rows = rows
        self.train_loader = train_loader
        self.save_hyperparameters(ignore= ["modality","encoder","test_loader","decoder","crop_rate"])
        self.loss_func = nn.MSELoss(reduction="sum")


    def get_mod_cropped(self,x):
        
        
        rows = self.rows  
        
        x[:,:,rows:,:] = 0.0
        return x
    
    def training_step(self, x) :
        
        self.train()
      
        x = self.get_mod_cropped(x[0])
        batch_size = x.size(0)
        recon ,z  = self.forward(x)  
        
        regularization = 0.0
        if self.regularization != None:
            if self.regularization == "l1":
                regularization = torch.abs(z).sum() 
            elif self.regularization == "l2":
                regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x,recon)
        total_loss= recon_loss + self.alpha * regularization
        
        self.logger.experiment.add_scalar("loss/train", total_loss/batch_size, self.global_step)
        
        return{"loss":total_loss / batch_size, "recon_loss": recon_loss.detach() / batch_size, "regularization": regularization / batch_size} 




    def test_step(self, x, batch_idx):
        
        x = self.get_mod_cropped(x[0])
        batch_size = x.size(0)
        recon ,z  = self.forward(x)  
        
        regularization = 0.0
        if self.regularization != None:
            if self.regularization == "l1":
                regularization = torch.abs(z).sum() 
            elif self.regularization == "l2":
                regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x,recon)
        total_loss= recon_loss + self.alpha * regularization
        
        self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
        
        return{"loss":total_loss/batch_size} 


    def validation_step(self, x, batch_idx):
        x = self.get_mod_cropped(x[0])
        batch_size = x.size(0)
        recon ,z  = self.forward(x)  
        
        regularization = 0.0
        if self.regularization != None:
            if self.regularization == "l1":
                regularization = torch.abs(z).sum() 
            elif self.regularization == "l2":
                regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x,recon)
        total_loss= recon_loss + self.alpha * regularization
        self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
        return{"loss":total_loss/batch_size}



    def on_train_epoch_end(self, *arg, **kwargs):
        
        if self.current_epoch % 5 ==0:
            self.encoder.eval()
            self.decoder.eval()
            test_batch =  self.get_mod_cropped( self.test_loader.to(self.device) )
            train_batch =self.get_mod_cropped(  self.train_loader.to(self.device) )
            print("Doing reconstruction")
            with torch.no_grad():
                recon, z = self.forward(test_batch)
                recon_train, z_train = self.forward(train_batch)
            
            print("test : std  : " + str(z.std().detach()) + "  mean : " +str(z.mean().detach()) )
            print("train : std  : " + str(z_train.std().detach()) + "  mean : " +str(z_train.mean().detach()) )
            
            log_modalities(self.logger, {self.modality:test_batch}, [self.modality], self.current_epoch ,prefix="real_test/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality:recon }, [self.modality], self.current_epoch ,prefix="recon_test/" ,nb_samples=8)
            
            log_modalities(self.logger, {self.modality:train_batch}, [self.modality], self.current_epoch ,prefix="real_train/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality:recon_train }, [self.modality], self.current_epoch ,prefix="recon_train/" ,nb_samples=8)
      
          

    def encode(self, x):
        return self.encoder(x)


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        z = self.encode(x)
        return self.decode(z) ,z


    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr , betas=(0.9,0.999), weight_decay=self.decay
                                     ,amsgrad=True 
                                    )
        return optimizer
    
    def reconstruction_loss(self, x,recon):
        
        return   self.loss_func(x,recon)



if __name__ =="__main__":
    
    args = parser.parse_args()
    Batch_size = 64
    NUM_epoch =100
    rows = [28,0,5,10,15,20,25]
    train_l, test_l = get_mnist_dataset(batch_size= Batch_size)
    train_samp = next(iter(train_l))[0][:8,]
    test_samp = next(iter(test_l))[0][:8,]
    r =args.rows
    ae =AE("mnist",train_loader= train_samp,test_loader=test_samp,
            enc = MnistEncoder(latent_dim=16,deterministic=True),
            dec= MnistDecoder(latent_dim=16),rows=r,lr=1e-3)
        
    CHECKPOINT_DIR = "runs/trained_models/ae_mnist_rows/"
  
    tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                    name="crop_"+str(r)
                                    )
    
   
    trainer = pl.Trainer(
                logger=tb_logger,
                check_val_every_n_epoch=25,
                accelerator='gpu',
                devices=1,
                max_epochs=NUM_epoch,
                default_root_dir=CHECKPOINT_DIR
            )

    trainer.fit(model=ae, train_dataloaders=train_l,
                        val_dataloaders=test_l)

