
import argparse




def get_config():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--task_0', type=int, default=0)
    parser.add_argument('--task_n', type=int, default=-1 )

    
    parser.add_argument('--type',  type=str, default="c",help='conditional or joint version of MINDE' )

    #parser.add_argument('--use_pretrain',  type=bool, default=False )
    parser.add_argument('--preprocessing',  type=str, default="rescale" )
    parser.add_argument('--arch',  type=str, default="mlp",help='architecture of the model, either "tx" for a transformer or "mlp"' )
    parser.add_argument('--Train_Size',  type=int, default=100000 )
    parser.add_argument('--Test_Size',  type=int, default=10000 )
    parser.add_argument('--seed',  type=int, default=0 )
          


    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of epochs before logging the results')

    # General Model settings

    parser.add_argument('--max_epochs', type=int, default=250,
                        help='number of epochs for training, in appedix of the paper we do an ablation study on this parameter')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate used for training')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size used for training')
    parser.add_argument('--test_epoch', type=int, default=1,
                        help='number of epochs between each evaluation of the model on the test set')
    
    parser.add_argument('--mc_iter', type=int, default=10,
                        help='NB MC iterations to estimate the O-information')
    
    parser.add_argument('--use_ema', action="store_true",
                        help='If true use Exponential Moving Average durring training and evaluation')
    
    parser.add_argument('--importance_sampling', action="store_true",
                        help='if true use importance sampling to train and  estimate the O-information')
    
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma value')
    
    parser.add_argument('--debug', action="store_true",
                        help='Debug option will log estimates of entropies durring training')
    # General GPU options

    parser.add_argument('--nb_workers', type=int, default=8,
                        help='Nb workers for the dataloader')
    parser.add_argument('--devices', type=int, default=1,
                        help='gpu device to use')
    parser.add_argument('--accelerator', type=str,
                        default="gpu", help='gpu or cpu')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=50,
                        help='number of epochs between each checkpointing')
    parser.add_argument('--out_dir', type=str, default="logs/trained_models/",
                        help='Where to store the model logs and checkpoints" ')
    parser.add_argument('--results_dir', type=str, default="results/",
                        help='Where to store the results')
    parser.add_argument('--benchmark', type=str, default="benchmark_name",
                        help='The name of the benchmark')
    return parser

