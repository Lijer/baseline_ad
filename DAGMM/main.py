# code based on https://github.com/danieltan07

import numpy as np
import argparse 
import torch

from train import TrainerDAGMM
from test import eval
from preprocessData import get_KDDCup99
from util import writeResults_my


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="number of epochs")           
    parser.add_argument("--data_dir", type=str, default='./data/thyroid.mat',
                        help="data path")           
    parser.add_argument("--c_in", type=int, default=6,
                        help="input dimension")     
    parser.add_argument("--h", type=int, default=2,
                        help="h dimension")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.005,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    #parsing arguments.
    args = parser.parse_args() 
    print("*"*10, " "*2, args.data_dir, " "*2, "*"*10)
    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get train and test dataloaders.
    data = get_KDDCup99(args)
    ROCAUC_list = []
    ap_list = []
    for i in range(10):
        DAGMM = TrainerDAGMM(args, data, device, args.c_in, args.h)
        DAGMM.train()
        ROCAUC, ap = eval(DAGMM.model, data, device, 4) # data[1]: test dataloader
        ROCAUC_list.append(ROCAUC)
        ap_list.append(ap)
    print(ap_list)
    print('avg:',np.array(ap_list).mean())
    print('std:',np.array(ap_list).std()) 
    print(ROCAUC_list)
    print('avg:',np.array(ROCAUC_list).mean())
    print('std:',np.array(ROCAUC_list).std())
    writeResults_my(args.data_dir, np.array(ROCAUC_list).mean(), np.array(ROCAUC_list).std(), np.array(ap_list).mean(), np.array(ap_list).std(),)