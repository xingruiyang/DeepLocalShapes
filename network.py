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


class PointNetFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels=3,
                 feat_size=1024,
                 layer_dims=[64, 128],
                 global_feat=True,
                 activation=F.relu,
                 batchnorm=True,
                 activation_last=False,
                 transposed_input=False):
        super(PointNetFeatureExtractor, self).__init__()

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)

        self.conv_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                              layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm
        self.activation_last = activation_last
        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        r"""Forward pass through the PointNet feature extractor.
        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.
        """
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).
        if self.batchnorm:
            x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        else:
            x = self.activation(self.conv_layers[0](x))

        # Pass through the remaining layers (until the penultimate layer).
        for idx in range(1, len(self.conv_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))

        if self.batchnorm:
            x = self.bn_layers[-1](self.conv_layers[-1](x))
        else:
            x = self.conv_layers[-1](x)

        if self.activation_last:
            x = self.activation(x)

        if self.global_feat:
            # Max pooling.
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, self.feat_size)

        return x


class PointNetTransformer(nn.Module):
    def __init__(self, in_channels=3):
        super(PointNetTransformer, self).__init__()

        self.in_channels = in_channels

        self.feature_extractor = PointNetFeatureExtractor(
            in_channels=in_channels, feat_size=1024,
            layer_dims=[64, 128], global_feat=True,
            activation=F.relu, batchnorm=True,
            activation_last=True,
            transposed_input=True
        )

        self.layers = nn.Sequential(
            self.feature_extractor,
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.init_params()

    def forward(self, x, transpose_input=False):
        if transpose_input:
            x = torch.transpose(x, 1, 2)
        assert x.size(1) == 3

        mat = self.gram_schmidt(self.layers(x))
        # mat = self.to_rot_mat(self.layers(x))

        return mat

    def gram_schmidt(self, rots):
        v1 = rots[..., :3]
        v1 = v1 / torch.max(torch.sqrt(torch.sum(v1**2, dim=-1, keepdim=True)),
                            torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
        v2 = rots[..., 3:] - \
            torch.sum(v1 * rots[..., 3:], dim=-1, keepdim=True) * v1
        v2 = v2 / torch.max(torch.sqrt(torch.sum(v2**2, dim=-1, keepdim=True)),
                            torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
        v3 = v1.cross(v2)

        rots = torch.stack([v1, v2, v3], dim=2)

        return rots

    def init_params(self):
        self.layers[-1].weight.data.zero_()
        self.layers[-1].bias.data.copy_(torch.tensor([1,
                                        0, 0, 0, 1, 0], dtype=torch.float))

    @classmethod
    def create_from_ckpt(self, ckpt, device=torch.device('cpu')):
        transformer = PointNetTransformer()
        data = torch.load(ckpt, map_location=device)
        transformer.load_state_dict(data["model_state_dict"])
        return transformer
