import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitNet(nn.Module):
    def __init__(self,
                 d_in,
                 dims):
        super(ImplicitNet, self).__init__()

        dims = [d_in] + dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, "lin_"+str(layer), lin)

        # self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        # self.activation = nn.Softplus(beta=100)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return self.tanh(x)
        # return x
