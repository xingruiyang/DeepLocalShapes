import json
import math

import numpy as np
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


class ImplicitNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=True,
                 clamp_dist=-1,
                 act_fn="leaky_relu",
                 optimizer="Adam",
                 freeze_decoder=False):
        super(ImplicitNet, self).__init__()

        dims = [latent_dim+3] + hidden_dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
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
        self.latent_vecs = None
        self.latent_dim = latent_dim
        self.freeze_decoder = freeze_decoder

    def forward(self, inputs):
        x = inputs
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act_fn(x)
        return self.tanh(x) if self.use_tanh else x

    def initialize_latents(self, num_latents=0, ckpt=None, device=torch.device('cpu')):
        if ckpt is None:
            self.latent_vecs = torch.zeros(
                (int(num_latents), self.latent_dim)).to(device)
            torch.nn.init.normal_(self.latent_vecs, 0, 0.01**2)
            self.latent_vecs.requires_grad_()
        else:
            latent_vecs = np.load(ckpt)
            self.latent_vecs = torch.from_numpy(latent_vecs).to(device)

    def configure_optimizers(self):
        if self.freeze_decoder:
            trainable_params = [{
                'params': self.latent_vecs, 'lr': 1e-3}]
        else:
            trainable_params = [{
                'params': self.parameters(), 'lr': 1e-3},
                {'params': self.latent_vecs, 'lr': 1e-3}]
        optimizer = self.optimizer(trainable_params)
        return optimizer

    def training_step(self, train_batch):
        index, pnts, gt_sdf = train_batch

        if self.use_tanh:
            gt_sdf = torch.tanh(gt_sdf)

        latents = torch.index_select(self.latent_vecs, 0, index.int())
        inputs = torch.cat([latents, pnts], dim=-1)

        pred_sdf = self.forward(inputs).squeeze()

        if self.clamp:
            gt_sdf = torch.clamp(gt_sdf, -self.clamp_dist, self.clamp_dist)
            pred_sdf = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)

        sdf_loss = (gt_sdf-pred_sdf).abs().mean()
        latent_loss = latents.abs().mean()
        loss = sdf_loss + 1e-3 * latent_loss

        return {
            'loss': loss,
            'sdf_loss': sdf_loss,
            'latent_loss': latent_loss
        }

    def save_ckpt(self, filename, epoch=0):
        model_state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict()}
        torch.save(model_state_dict, filename)

    def save_latents(self, filename):
        latent_vecs = self.latent_vecs.detach().cpu().numpy()
        np.save(filename, latent_vecs)

    @staticmethod
    def create_from_cfg(cfg, ckpt=None, device=torch.device('cpu')):
        net_args = json.load(open(cfg, 'r'))
        network = ImplicitNet(**net_args['params']).to(device)
        if ckpt is not None:
            data = torch.load(ckpt, map_location=device)
            network.load_state_dict(data["model_state_dict"])
        return network
