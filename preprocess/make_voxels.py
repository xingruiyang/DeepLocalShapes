import argparse
import os
import pickle

import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree

from transformer import PointNetTransformer
from utils import load_model


def toldi_compute_xyz(surf, centroid):
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


class Voxelizer(object):
    def __init__(self, pcd, network=None, mnfld_pnts=4096) -> None:
        super().__init__()

        if isinstance(pcd, str):
            pcd = np.load(pcd)
        self.points = pcd[:, 1:4]
        self.normals = pcd[:, 4:]
        self.weights = pcd[:, 0]

        self.mnfld_pnts = mnfld_pnts
        self.network = network
        if network is not None:
            self.network = PointNetTransformer()
            load_model(network, self.network)
            self.network.cuda().eval()

    def display_sdf(self, pts, sdf):  # , voxel_size, rotation, centroid):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # bbox = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=[-1.5]*3, max_bound=[1.5]*3)
        o3d.visualization.draw_geometries([pcd])

    def get_rotation(self, pts, centroid, voxel_size):
        volume_surface = (pts - centroid) / (1.5 * voxel_size)
        volume_surface = volume_surface[None, ...].float()
        orientation = self.network(
            volume_surface, transpose_input=True)
        return orientation[0, ...]

    def create_voxels(self, voxel_size, out_path=None):
        points = torch.from_numpy(self.points).cuda()
        normals = torch.from_numpy(self.normals).cuda()
        weights = torch.from_numpy(self.weights).cuda()
        rand_pts = points * (0.96-torch.rand(points.shape[0], 1).cuda() * 0.4)

        voxels = torch.div(points, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        print("{} voxels to be sampled".format(voxels.shape[0]))

        num_rand_pts = min(2**17, self.points.shape[0])
        surface = self.points[np.random.permutation(
            self.points.shape[0])[:num_rand_pts], :]
        kd_tree = KDTree(surface)
        rand_sdf = kd_tree.query(rand_pts.detach().cpu().numpy())
        rand_sdf = torch.from_numpy(rand_sdf[0][:, 0])
        print("constructed kd-tree")

        dist, ind = kd_tree.query(voxels.detach().cpu().numpy())
        voxels = surface[ind, :].squeeze()
        voxels = torch.from_numpy(voxels).cuda()

        # d1 = torch.rand((points.shape[0], 1)).cuda() * 0.02
        # d1 = 0.015
        # neg_points = points - d1 * normals
        # pos_points = points + d1 * normals
        # neg_sdf = torch.zeros((points.shape[0],1)).cuda() - d1
        # pos_sdf = torch.zeros((points.shape[0],1)).cuda() + d1
        # print(d1)

        # self.display_sdf(torch.cat([neg_points, pos_points, rand_pts], 0).detach().cpu().numpy(),
        #                  torch.cat([neg_sdf, pos_sdf, rand_sdf], 0).detach().cpu().numpy().squeeze())

        max_num_pcd = 2**12
        samples = []
        centroids = []
        rotations = []
        num_aligned = 0
        for vid in range(voxels.shape[0]):
            print("{}/{}".format(vid, voxels.shape[0]))
            voxel = voxels[vid, :]
            pcd = points - voxel
            dist = torch.norm(pcd, p=np.inf, dim=-1)
            selector = dist < (1.5 * voxel_size)
            pcd = pcd[selector, :]
            normal = normals[selector, :]
            weight = weights[selector]
            # neg = (neg_points-voxel)[selector, :]
            # pos = (pos_points-voxel)[selector, :]
            # nsdf = neg_sdf[selector, :]
            # psdf = pos_sdf[selector, :]

            if pcd.shape[0] > 2048:
                surf = pcd[torch.randperm(pcd.shape[0])[:2048], :]
            else:
                surf = pcd
            # surf = pcd
            centroid = torch.mean(surf, dim=0)
            rotation = np.eye(3)
            if self.network is not None and surf.shape[0] >= 2048:
                surf = surf[torch.randperm(pcd.shape[0])[:2048], :]
                rotation = self.get_rotation(surf, centroid, voxel_size)
            elif surf.shape[0] > 10:
                rotation = toldi_compute_xyz(
                    surf.detach().cpu().numpy(), centroid.detach().cpu().numpy())
                num_aligned += 1

            centroids.append(centroid.detach().cpu().numpy())
            rotations.append(rotation)

            if pcd.shape[0] > max_num_pcd:
                rand_sel = torch.randperm(pcd.shape[0])[:max_num_pcd]
                pcd = pcd[rand_sel, :]
                normal = normal[rand_sel, :]
                weight = weight[rand_sel]

            rand_sample = rand_pts - voxel
            dist = torch.norm(rand_sample, p=np.inf, dim=-1)
            selector = dist < (1.5 * voxel_size)
            rand_sample = rand_sample[selector, :]
            rand_weight = weights[selector]
            rand_sdf_sample = rand_sdf[selector]

            # global_pts = (torch.rand((512, 3)) * 3 - 1.5).cuda() * voxel_size
            # global_sdf = torch.ones((512, 1)).cuda() * -1
            # global_weight = torch.zeros((512, )).cuda()
            # global_pts = torch.matmul(
            #     global_pts.cuda(), torch.from_numpy(rotation).float().cuda()) + centroid

            d = 0.015
            d2 = 0.005
            d = torch.ones((pcd.shape[0], 1)) * d
            sample_pts = torch.cat([
                pcd,
                pcd + normal*d.cuda(),
                pcd - normal*d.cuda(),
                # pcd + normal*d2,
                # pcd - normal*d2,
                rand_sample,
                # global_pts
            ], axis=0)
            sample_sdf = torch.cat([
                torch.zeros((pcd.shape[0],)),
                torch.zeros((pcd.shape[0],))+d[..., 0],
                torch.zeros((pcd.shape[0],))-d[..., 0],
                # torch.zeros((pcd.shape[0],))+d2,
                # torch.zeros((pcd.shape[0],))-d2,
                rand_sdf_sample,
                # global_sdf
            ], axis=0)
            sample_weights = torch.cat([
                weight,
                weight,
                weight,
                # weight,
                # weight,
                rand_weight,
                # global_weight
            ], axis=0)

            # sample_pts = torch.cat([
            #     pcd,
            #     pos,
            #     neg,
            #     rand_sample,
            #     global_pts
            # ], dim=0)
            # sample_sdf = torch.cat([
            #     torch.zeros((pcd.shape[0],1)).cuda(),
            #     psdf,
            #     nsdf,
            #     rand_sdf_sample,
            #     global_sdf
            # ], dim=0)
            # sample_weights = torch.cat([
            #     weight,
            #     weight,
            #     weight,
            #     rand_weight,
            #     global_weight
            # ], dim=0)

            # d = 0.015
            # d = torch.randn((pcd.shape[0], 1)) * d
            # sample_pts = torch.cat([
            #     # pcd,
            #     pcd + normal*d.cuda(),
            #     rand_sample,
            #     global_pts.cuda()
            # ], axis=0)
            # sample_sdf = torch.cat([
            #     # torch.zeros((pcd.shape[0],)),
            #     torch.zeros((pcd.shape[0],))+d[..., 0],
            #     rand_sdf_sample,
            #     global_sdf
            # ], axis=0)
            # sample_weights = torch.cat([
            #     # weight,
            #     weight,
            #     rand_weight,
            #     global_weight.cuda()
            # ], axis=0)

            # self.display_sdf(sample_pts.detach().cpu().numpy(),
            #                  sample_sdf.detach().cpu().numpy().squeeze())

            # d = torch.rand((pcd.shape[0], 1)) * 0.03 - 0.015
            # sample_pts = torch.cat([
            #     # pcd,
            #     pcd + normal*d.cuda(),
            #     rand_sample,
            #     global_pts.cuda(),
            # ], axis=0)
            # sample_sdf = torch.cat([
            #     # torch.zeros((pcd.shape[0],)),
            #     torch.zeros((pcd.shape[0],))+d[..., 0],
            #     rand_sdf_sample,
            #     global_sdf
            # ], axis=0)
            # sample_weights = torch.cat([
            #     # weight,
            #     weight,
            #     rand_weight,
            #     global_weight.cuda()
            # ], axis=0)

            sample = np.zeros((sample_pts.shape[0], 6))
            sample[:, 0] = vid
            sample[:, 1:4] = sample_pts.detach().cpu().numpy()
            sample[:, 4] = sample_sdf.detach().cpu().numpy().squeeze()
            sample[:, 5] = sample_weights.detach().cpu().numpy()
            samples.append(sample)

        samples = np.concatenate(samples, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)
        voxels = voxels.detach().cpu().numpy()

        if out_path is None:
            print("{} points sampled with {} voxels aligned".format(
                samples.shape[0], num_aligned))
        else:
            with open(os.path.join(out_path, 'meta.txt'), 'w') as f:
                f.write("{} points sampled with {}/{} (aligned) voxels".format(
                    samples.shape[0], num_aligned, voxels.shape[0]))
                f.close()
        return samples, voxels, centroids, rotations, surface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--mnfld-pnts', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    args = parser.parse_args()

    voxelizer = Voxelizer(args.pcd, args.network, args.mnfld_pnts)
    samples = voxelizer.create_voxels(args.voxel_size)
    samples, voxels, centroids, rotations, surface = samples

    sample_name = 'samples.npy'
    surface_pts_name = 'surface_pts.npy'
    surface_sdf_name = 'surface_sdf.npy'

    out = dict()
    out['samples'] = sample_name
    out['surface_pts'] = surface_pts_name
    out['surface_sdf'] = surface_sdf_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
    np.save(os.path.join(args.output, surface_pts_name),
            surface.astype(np.float32))
