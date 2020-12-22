import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DAGMM
from forward_step import ComputeLoss
from utils.utils import weights_init_normal

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device, c_in, h):
        self.args = args
        self.c_in = c_in
        self.train_loader, self.test_loader = data
        self.device = device
        self.h = h


    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim, self.c_in, self.h).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.device, self.args.n_gmm)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            # for x, _ in Bar(self.train_loader):
            # print(type(self.train_loader))
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                _, x_hat, z, gamma = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                # for param in self.model.parameters():
                #     print(param)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
                

        

