import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

activations = {
    "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    "soft_plus": nn.Softplus(beta=100)
}

optimizers = {
    "Adam": torch.optim.Adam
}


class ImplicitNet(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=False,
                 input_scale=-1,
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
        self.input_scale = input_scale
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

    def check_latent_vecs(self):
        if self.latent_vecs is None:
            raise ValueError(
                "latent vectors are not initialized.\ncall 'initialize_latents' first")

    def initialize_latents(self, num_latents):
        self.latent_vecs = torch.zeros((num_latents, self.latent_dim))
        torch.nn.init.normal_(self.latent_vecs, 0, 0.01**2)

    def configure_optimizers(self):
        self.check_latent_vecs()

        if self.freeze_decoder:
            optim_params = [self.latent_vecs]
        else:
            optim_params = [{
                'params': self.parameters()},
                {'params': self.latent_vecs}]

        return self.optimizer(optim_params, lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        loss, sdf_loss, latent_loss = self.compute_loss(train_batch)

        self.log('train/sdf_loss', sdf_loss.item())
        self.log('train/latent_loss', latent_loss.item())

        return loss

    def train_dataloader(self):
        pass

    def compute_loss(self, batch):
        latent_ind, points, sdf_values, \
            weights, centroids, rotations = batch
        latents = torch.index_select(
            self.latent_vecs, 0, latent_ind)

        if centroids is not None:
            points -= centroids

        if rotations is not None:
            points = torch.bmm(
                points.unsqueeze(1),
                rotations.transpose(1, 2)).squeeze()

        if self.input_scale > 0:
            points *= self.input_scale
        points = torch.cat([latents, points], dim=-1)
        sdf_pred = self.forward(points).squeeze()

        if self.use_tanh:
            sdf_values = torch.tanh(sdf_values)

        if self.clamp_dist > 0:
            sdf_pred = torch.clamp(
                sdf_pred, -self.clamp_dist, self.clamp_dist)
            sdf_values = torch.clamp(
                sdf_values, -self.clamp_dist, self.clamp_dist)

        sdf_loss = (((sdf_values-sdf_pred)*weights).abs()).mean()
        latent_loss = latents.abs().mean()
        loss = sdf_loss + latent_loss * 1e-3

        return loss, sdf_loss, latent_loss


if __name__ == '__main__':
    decoder = ImplicitNet(125, [128, 128, 128])
    decoder.initialize_latents(100)
    trainer = pl.Trainer()
    train_loader = None
    trainer.fit(decoder, train_loader)
