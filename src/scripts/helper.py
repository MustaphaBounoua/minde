from torch.utils.data import Dataset ,DataLoader
import torch
from sklearn import preprocessing
from src.scripts.config import get_config
import numpy as np


def get_default_config():
    return get_config().parse_args([])

BATCH_SIZE = [128,256]
BATCH_SIZE_TEST = 1000

class SynthetitcDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        self.x = data [0]
        self.y = data [1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x":torch.FloatTensor(self.x[idx]) ,"y":torch.FloatTensor(self.y[idx]) }
    


def get_data_loader(args, task,batch_size= None):


    

    size_train = args.Train_Size
    size_test = args.Test_Size

    X, Y = task.sample(size_test+size_train, seed=args.seed)
    if args.preprocessing == "rescale":
        X = preprocessing.StandardScaler(copy=True).fit_transform(X)
        Y = preprocessing.StandardScaler(copy=True).fit_transform(Y)
    
    
    x_train , y_train =  X[:size_train,], Y[:size_train,]
    x_test , y_test =  X[size_train:,], Y[size_train:,]
    

    data_train, data_test =  [x_train , y_train], [x_test , y_test]


    train, test = SynthetitcDataset(data_train),SynthetitcDataset(data_test)
    
    if batch_size == None:
        if task.dim_x <=5:
            batch_size = BATCH_SIZE[0]
        elif task.dim_x > 5:
            batch_size = BATCH_SIZE[1]


    batch_size_test = BATCH_SIZE_TEST

    train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,pin_memory =True,
                          num_workers=32, drop_last=True)

    test_loader = DataLoader(test, batch_size=batch_size_test,
                          shuffle= False,
                          num_workers=32, drop_last=False,pin_memory =True
                          )
    

    return train_loader,test_loader




def get_samples(test_loader , n_sample, device ="cuda"):
   

    X =torch.Tensor().to(device)
    Y =torch.Tensor().to(device)

    for batch in test_loader:
        X = torch.cat([X,batch["x"].to(device)])
        Y = torch.cat([Y,batch["y"].to(device)])
       
    return  {
    "x":X[:n_sample,],
    "y":Y[:n_sample,]
  
    }
