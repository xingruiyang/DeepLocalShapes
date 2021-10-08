import argparse
import os
import pickle

import numpy as np
import torch
import trimesh
from sklearn.neighbors import KDTree


class Voxelizer(object):
    def __init__(self, pcd_path, mnfld_pnts=4096) -> None:
        super().__init__()
        raw_data = np.load(pcd_path)
        self.points = raw_data[:, 1:4]
        self.normals = raw_data[:, 4:]
        self.weights = raw_data[:, 0]
        self.mnfld_pnts = mnfld_pnts

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def create_voxels(self, voxel_size):
        points = torch.from_numpy(self.points).cuda()
        normals = torch.from_numpy(self.normals).cuda()
        weights = torch.from_numpy(self.weights).cuda()
        rand_pts = points * (1-torch.rand(points.shape[0], 1).cuda() * 0.4)

        voxels = torch.div(points, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        print("{} voxels to be sampled".format(voxels.shape[0]))

        surface = self.points[np.random.permutation(
            self.points.shape[0])[:2**16], :]
        kd_tree = KDTree(surface)
        rand_sdf = kd_tree.query(rand_pts.detach().cpu().numpy())
        rand_sdf = torch.from_numpy(rand_sdf[0][:, 0])
        print("constructed kd-tree")

        max_num_pcd = 2**14
        samples = []
        centroids = []
        rotations = []
        for vid in range(voxels.shape[0]):
            print("{}/{}".format(vid, voxels.shape[0]))
            voxel = voxels[vid, :]
            pcd = points - voxel
            dist = torch.norm(pcd, p=np.inf, dim=-1)
            selector = dist < (1.5 * voxel_size)
            pcd = pcd[selector, :]
            normal = normals[selector, :]
            weight = weights[selector]

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

            d = torch.rand((pcd.shape[0], 1)) * 0.015
            sample_pts = torch.cat([
                # pcd,
                pcd + normal*d.cuda(),
                pcd - normal*d.cuda(),
                rand_sample], axis=0)
            sample_sdf = torch.cat([
                # torch.zeros((pcd.shape[0],)),
                torch.zeros((pcd.shape[0],))+d[..., 0],
                torch.zeros((pcd.shape[0],))-d[..., 0],
                rand_sdf_sample], axis=0)
            sample_weights = torch.cat([
                # weight,
                weight,
                weight,
                rand_weight
            ], axis=0)

            sample = np.zeros((sample_pts.shape[0], 6))
            sample[:, 0] = vid
            sample[:, 1:4] = sample_pts.detach().cpu().numpy()
            sample[:, 4] = sample_sdf.detach().cpu().numpy()
            sample[:, 5] = sample_weights.detach().cpu().numpy()
            samples.append(sample)
            centroids.append(np.zeros(3,))
            rotations.append(np.eye(3))

        samples = np.concatenate(samples, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)
        voxels = voxels.detach().cpu().numpy()

        print("{} points sampled with {} voxels aligned".format(
            samples.shape[0], rotations.shape[0]))

        return samples, voxels, centroids, rotations, surface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--mnfld-pnts', type=int, default=4096)
    args = parser.parse_args()

    voxelizer = Voxelizer(args.pcd, args.mnfld_pnts)
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
