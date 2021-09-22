import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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

    def to_rot_mat(self, alpha):
        angle_axis = torch.zeros((alpha.shape[0], 2))
        if alpha.is_cuda:
            angle_axis = angle_axis.cuda()
        angle_axis = torch.cat([alpha, angle_axis], dim=-1)
        # angle_axis[:, 0] = alpha
        return angle_axis_to_rotation_matrix(angle_axis)[:, :3, :3]

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
