import sys
from utils import cal_xyz_axis
import trimesh
import torch
import numpy as np


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
# mesh = trimesh.primitives.Cylinder(raidus=1, height=2)
# trimesh.PointCloud(pnts).show()
scene = trimesh.Scene()
for i in range(3):
    for j in range(3):
        pnts = mesh.sample(500)
        pnts = torch.from_numpy(pnts)
        centroids = torch.mean(pnts, dim=0)
        ref_pnts = pnts-centroids
        rotation = cal_xyz_axis(ref_pnts)
        print(torch.det(rotation))
        # print(rotation)
        # rotation = cal_xyz_axis2(pnts.cpu().numpy(), centroids.cpu().numpy())
        # rotation = torch.from_numpy(rotation)

        pnts = torch.matmul(pnts, rotation.transpose(0, 1))
        pnts = trimesh.PointCloud(pnts)
        transform = np.eye(4)
        transform[0, 3] = i * 2
        transform[1, 3] = j * 2
        scene.add_geometry(pnts, transform=transform)
scene.show()
