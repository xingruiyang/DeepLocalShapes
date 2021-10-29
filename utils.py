import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch


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


def RodsRotatFormula(a, b):
    B, _ = a.shape
    device = a.device
    b = b.to(device)
    c = torch.cross(a, b)
    theta = torch.acos(F.cosine_similarity(a, b)).unsqueeze(1).unsqueeze(2)

    c = F.normalize(c, p=2, dim=1)
    one = torch.ones(B, 1, 1).to(device)
    zero = torch.zeros(B, 1, 1).to(device)
    a11 = zero
    a12 = -c[:, 2].unsqueeze(1).unsqueeze(2)
    a13 = c[:, 1].unsqueeze(1).unsqueeze(2)
    a21 = c[:, 2].unsqueeze(1).unsqueeze(2)
    a22 = zero
    a23 = -c[:, 0].unsqueeze(1).unsqueeze(2)
    a31 = -c[:, 1].unsqueeze(1).unsqueeze(2)
    a32 = c[:, 0].unsqueeze(1).unsqueeze(2)
    a33 = zero
    Rx = torch.cat(
        (torch.cat((a11, a12, a13), dim=2), torch.cat(
            (a21, a22, a23), dim=2), torch.cat((a31, a32, a33), dim=2)),
        dim=1)
    I = torch.eye(3).to(device)
    R = I.unsqueeze(0).repeat(B, 1, 1) + torch.sin(theta) * \
        Rx + (1 - torch.cos(theta)) * torch.matmul(Rx, Rx)
    return R.transpose(-1, -2)


def cal_xyz_axis(ref_pnts):
    z_axis = cal_z_axis(ref_pnts)
    x_axis = cal_x_axis(ref_pnts, z_axis)
    y_axis = torch.cross(z_axis, x_axis)
    return torch.stack([x_axis, y_axis, z_axis], dim=0)


def to_o3d(arr: np.ndarray, color: np.ndarray = None):
    """Converts NumPy pnts to open3d format
    Args:
        arr (ndarray|tensor): pnts of shape Nx3
        color (ndarray|tensor, optional): colors of shape Nx3
    """
    if isinstance(arr, torch.Tensor):
        arr = deepcopy(arr).detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = deepcopy(color).detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def draw_voxels(pnts: np.ndarray,
                voxels: np.ndarray,
                voxel_size: float = 0.1):
    """Visualize pnts with a set of voxels
    Args:
        pnts (ndarray|tensor): pnts of shape Nx3
        voxels (ndarray|tensor): voxels of shape Mx3
        voxels (float, optional): the length of voxels
    """
    if isinstance(pnts, torch.Tensor):
        pts = pnts.detach().cpu().numpy()
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()
    num_voxels = voxels.shape[0]
    geometries = []
    point_cloud = to_o3d(pts)
    for i in range(num_voxels):
        voxel = voxels[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        geometries.append(bbox_inner)
    o3d.visualization.draw_geometries(geometries+[point_cloud])


def display_sdf(pts: np.ndarray, sdf: np.ndarray, only_negative=False):
    """Visualize pnts with corresponding sdf values
    Args:
        pts (ndarray|tensor): pnts of shape Nx3
        sdf (ndarray|tensor): sdf values of shape Nx3
        only_negative (bool, optional): only show negative samples
    """

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(pts, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()

    if only_negative:
        pts = pts[sdf < 0, :]
        sdf = sdf[sdf < 0]

    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud])
