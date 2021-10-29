import sys
import trimesh
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform.rotation import Rotation as R


def normalize(vec: torch.Tensor):
    if len(vec.shape) < 3:
        vec_norm = torch.norm(vec, p=2)
        return vec / vec_norm if vec_norm != 0 else vec


def cal_z_axis(ref_points: torch.Tensor):
    '''Calculate the z-axis from point clouds
    Args:
        ref_points(Tensor): needs to be centred
    '''
    cov_mat = torch.matmul(ref_points.transpose(-1, -2), ref_points)
    _, eig_vecs = torch.linalg.eigh(cov_mat)
    z_axis = eig_vecs[:, 0]
    mask = (torch.sum(-ref_points * z_axis) < 0).float()
    return z_axis if mask > 0 else -z_axis


def cal_x_axis(ref_pnts, z_axis):
    '''Calculate the x-axis from point clouds
    Args:
        ref_points (tensor): needs to be centred
        z_axis (tensor): the estimated z_axis
    '''
    z_proj = torch.sum(ref_pnts*z_axis, dim=-1, keepdim=True)
    sign_weight = z_proj**2
    sign_weight[z_proj < 0] *= -1

    vec_proj = ref_pnts - z_proj * z_axis
    dist = torch.norm(ref_pnts, p=2, dim=-1, keepdim=True)
    supp = torch.max(dist)
    dist_weight = (supp - dist)**2

    x_axis = dist_weight * sign_weight * vec_proj
    x_axis = torch.sum(x_axis, dim=0)

    return normalize(x_axis)


def cal_xyz_axis(ref_pnts):
    '''Calculate the local reference frame of a point patch 
    Args:
        ref_points(Tensor): needs to be centred
    '''
    z_axis = cal_z_axis(ref_pnts)
    x_axis = cal_x_axis(ref_pnts, z_axis)
    y_axis = torch.cross(x_axis, z_axis)
    return torch.stack([x_axis, y_axis, z_axis], dim=0)


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


def cal_xyz_axis2(surf, centroid):
    pcd = surf - centroid
    dist = np.linalg.norm(pcd, ord=2, axis=-1)
    dist_ind = dist.argsort()
    pcd = pcd[dist_ind, :]
    dist = dist[dist_ind]

    cov_mat = np.matmul(pcd.transpose(), pcd)
    _, eig_vecs = np.linalg.eigh(cov_mat)
    z_axis = eig_vecs[:, 0]
    z_sign = 0.0
    for i in range(pcd.shape[0]):
        vec_x = 0-pcd[i, 0]
        vec_y = 0-pcd[i, 1]
        vec_z = 0-pcd[i, 2]
        sign = (vec_x * z_axis[0] + vec_y * z_axis[1] + vec_z * z_axis[2])
        z_sign += sign
    if z_sign < 0:
        z_axis *= -1
    z_proj = np.dot(pcd, z_axis)
    sign_weight = z_proj**2
    sign_weight[z_proj < 0] *= -1

    vec_proj = np.zeros((pcd.shape[0], 3))
    for i in range(pcd.shape[0]):
        vec_proj[i, 0] = pcd[i, 0] - z_proj[i] * z_axis[0]
        vec_proj[i, 1] = pcd[i, 1] - z_proj[i] * z_axis[1]
        vec_proj[i, 2] = pcd[i, 2] - z_proj[i] * z_axis[2]

    supp = np.max(dist)
    dist_weight = (supp - dist)**2
    x_axis = dist_weight[:, None] * sign_weight[:, None] * vec_proj
    x_axis = np.sum(x_axis, axis=0)
    x_axis /= np.linalg.norm(x_axis, ord=2)

    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    return rotation


mesh = trimesh.load(sys.argv[1]).dump(True)
transformer = PointNetTransformer.create_from_ckpt(sys.argv[2])
transformer.eval()
# mesh = trimesh.primitives.Cylinder(raidus=1, height=2)
# trimesh.PointCloud(pnts).show()
scene = trimesh.Scene()
for i in range(3):
    for j in range(3):
        pnts = mesh.sample(2048)
        pnts = torch.from_numpy(pnts).float()
        centroids = torch.mean(pnts, dim=0)
        ref_pnts = pnts-centroids
        ref_pnts = torch.matmul(ref_pnts, torch.from_numpy(
            R.random().as_matrix()).float())

        rotation = cal_xyz_axis(ref_pnts)
        ref_pnts = torch.matmul(ref_pnts, rotation.transpose(0, 1))

        rotation = transformer(
            ref_pnts[None, ...], transpose_input=True).reshape(3, 3)

        ref_pnts = torch.matmul(ref_pnts, rotation.transpose(0, 1))
        ref_pnts = trimesh.PointCloud(ref_pnts.detach().cpu())
        transform = np.eye(4)
        transform[0, 3] = i * 2
        transform[1, 3] = j * 2
        scene.add_geometry(ref_pnts, transform=transform)
scene.show()
