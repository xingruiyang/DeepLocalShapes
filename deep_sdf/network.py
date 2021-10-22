import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    "soft_plus": nn.Softplus(beta=100)
}

optimizers = {
    "Adam": torch.optim.Adam
}


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenModule(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6.,
                 is_first=False, use_bias=True,
                 activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class ImplicitNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=True,
                 voxel_size=1,
                 clamp_dist=-1,
                 act_fn="leaky_relu",
                 optimizer="Adam"):
        super(ImplicitNet, self).__init__()

        dims = [latent_dim+3] + hidden_dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            # is_first = True if layer == 0 else False
            # lin = SirenModule(dims[layer], out_dim, 30 if is_first else 1, is_first=is_first)
            setattr(self, "lin_"+str(layer), lin)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        self.act_fn = activations.get(
            act_fn, nn.ReLU())
        self.optimizer = optimizers.get(
            optimizer, torch.optim.Adam)

        self.clamp_dist = clamp_dist
        self.clamp = clamp_dist > 0
        self.input_scale = 1.0/voxel_size
        self.latent_vecs = None
        self.latent_dim = latent_dim

    def forward(self, inputs):
        x = inputs
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act_fn(x)
        return self.tanh(x) if self.use_tanh else x

    def save_model(self, filename, epoch=0):
        model_state_dict = {
            "epoch": epoch,
            "hparams": None,
            "model_state_dict": self.state_dict()}
        torch.save(model_state_dict, filename)

    @staticmethod
    def create_from_cfg(cfg, ckpt=None, device=torch.device('cpu')):
        net_args = json.load(open(cfg, 'r'))
        network = ImplicitNet(**net_args['params']).to(device)
        if ckpt is not None:
            data = torch.load(ckpt, map_location=device)
            network.load_state_dict(data["model_state_dict"])
        return network
