
import argparse


def get_default_config():
    return get_config().parse_args([])


def get_config():
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_0', type=int, default=0)
    parser.add_argument('--task_n', type=int, default=-1 )
    parser.add_argument('--weighted', type=bool, default=False)

    parser.add_argument('--importance_sampling_train', type=bool, default=True)
    parser.add_argument('--importance_sampling_test',  type=bool, default=False)
    parser.add_argument('--use_pretrain',  type=bool, default=False )
    parser.add_argument('--preprocessing',  type=str, default="rescale" )
    parser.add_argument('--arch',  type=str, default=True )
    parser.add_argument('--N_test,  type=int, default=100000 )
    parser.add_argument('--N_test',  type=int, default=10000 )
    parser.add_argument('--seed',  type=int, default=0 )
            





    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of epochs before logging the results')
    parser.add_argument('--transformation', type=str, default="",
                        help='transformation applied to the data, either "H-C"  or "CDF" ')

    # General Model settings

    parser.add_argument('--o_inf_order', type=int, default=1,
                        help='Order of the denoising score functions to estimate. if Set to 1 learn the joint, marginals and conditionals necessary to compute O-information')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='number of epochs for training, in appedix of the paper we do an ablation study on this parameter')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate used for training')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size used for training')
    parser.add_argument('--weight_s_functions', action="store_true",
                        help='if set to 1, the model will assign a weight_s_functions to each denoising score function durring training, for instance joint score function will be selectly more likly than marginals')
    parser.add_argument('--test_epoch', type=int, default=250,
                        help='number of epochs between each evaluation of the model on the test set')
    parser.add_argument('--arch', type=str, default="tx",
                        help='architecture of the model, either "tx" for a transformer or "mlp"')
    parser.add_argument('--mc_iter', type=int, default=10,
                        help='NB MC iterations to estimate the O-information')
    parser.add_argument('--use_ema', action="store_true",
                        help='If true use Exponential Moving Average durring training and evaluation')
    parser.add_argument('--importance_sampling', action="store_true",
                        help='if true use importance sampling to train and  estimate the O-information')
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
    parser.add_argument('--results_dir', type=str, default="reprod_results/",
                        help='Where to store the results')
    
    return parser


def get_config_baseline():
    parser = argparse.ArgumentParser()

    # General benchmark settings
    parser.add_argument('--benchmark', type=str, default="red",
                        help='The benchmark to use either "red" for redundancy, "syn" for synergy or "mix" for a mix of both')
    parser.add_argument('--rho', type=float, default=0.1,
                        help='rho the correlation coefficient controlling the redundancy strenth')
    parser.add_argument('--seed', type=int, default=3,
                        help='seed for the experiment')
    parser.add_argument('--setting', type=int, default=0,
                        help='setting of the experiment either sytems of [3, 3, 4], [3, 3], [3]')
    parser.add_argument('--dim', type=int, default=1,
                        help='dimension of each variable in the system')
    parser.add_argument('--N', type=int, default=100*1000,
                        help='number of samples used for training')
    parser.add_argument('--N_test', type=int, default=100*1000,
                        help='number of samples used for testing')
    parser.add_argument('--transformation', type=str, default="",
                        help='transformation applied to the data, either "H-C" or "Spiral" or "CDF" ')
    # General Model settings

    parser.add_argument('--mi_e', type=str, default="MINE",help="MI estimator to use either 'MINE' or 'NWJ' or 'InfoNCE' or 'CLUB' ")
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--test_epoch', type=int, default=10,
                        help='number of epochs between each evaluation of the model on the test set')

    # General training options
    parser.add_argument('--nb_workers', type=int, default=16,
                        help='Nb workers for the dataloader')
    parser.add_argument('--devices', type=int, default=1,
                        help='gpu device to use')
    parser.add_argument('--accelerator', type=str,
                        default="gpu", help='gpu or cpu')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=50,
                        help='number of epochs between each checkpointing')
    parser.add_argument('--out_dir', type=str, default="trained_models",
                        help='Where to store the model logs and checkpoints" ')
    parser.add_argument('--results_dir', type=str, default="results",
                        help='Where to store the results')
    return parser


