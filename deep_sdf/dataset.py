import open3d as o3d
import time
import argparse
from sklearn.neighbors import KDTree
import torch
import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 orient: bool = False,
                 crop: bool = False,
                 training: bool = True):
        super(SampleDataset, self).__init__()
        self.root_path = data_path
        self.training = training

        raw_data = self.load_pickle('samples.pkl')
        self.voxels = raw_data['voxels']
        self.voxel_size = raw_data['voxel_size']

        self.num_latents = self.voxels.shape[0]
        self.samples = None
        self.surface = None
        self.rotations = None
        self.centroids = None

        if training:
            train_data = os.path.join(data_path, raw_data['samples'])
            self.samples = np.load(train_data)
        if crop:
            eval_data = os.path.join(data_path, raw_data['surface_pts'])
            self.surface = np.load(eval_data)
        if orient:
            self.rotations = raw_data.get('rotations', None)
            self.centroids = raw_data.get('centroids', None)

    def load_pickle(self, filename):
        with open(os.path.join(self.root_path, filename), "rb") as f:
            return pickle.load(f)

    def __len__(self):
        if self.training:
            return self.samples.shape[0]
        else:
            return self.num_latents

    def __getitem__(self, index):
        if self.training:
            indices = self.samples[index, 0].astype(int)
            points = self.samples[index, 1:4].astype(np.float32)
            sdf = self.samples[index, 4].astype(np.float32)
            weight = self.samples[index, 5].astype(np.float32)
            return indices, points, sdf, weight
        else:
            return index, self.voxels[index, ...].astype(np.float32)


class SceneDataset(Dataset):
    def __init__(
            self,
            path,
            max_sample=4096,
            training=False) -> None:
        super().__init__()

        meta = np.load(os.path.join(path, 'meta.npz'))
        self.voxels = meta['voxels']
        self.rotations = meta['rotations']
        self.centroids = meta['centroids']
        self.voxel_size = meta['voxel_size'].item()
        self.num_latents = self.voxels.shape[0]
        self.max_sample = max_sample
        self.training = training

        cloud = np.load(os.path.join(path, 'cloud.npy'))
        num_rand_pts = min(2**17, cloud.shape[0])
        self.surface = cloud[np.random.permutation(
            cloud.shape[0])[:num_rand_pts], 1:4]

        if self.training:
            start = time.time()
            print("begin preparing dataset {}".format(path))
            self.samples = self.prepare_dataset(cloud)
            print("{} points sampled, time elapsed {} s".format(
                self.samples.shape[0], time.time()-start))

    def __len__(self):
        if self.training:
            return self.samples.shape[0]
        else:
            return self.num_latents

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])

    def prepare_dataset(self, cloud, use_gpu=True):
        device = torch.device('cuda' if use_gpu else 'cpu')
        points = torch.from_numpy(cloud[:, 1:4]).to(device)
        normals = torch.from_numpy(cloud[:, 4:]).to(device)
        weights = torch.from_numpy(cloud[:, 0]).to(device)
        voxels = torch.from_numpy(self.voxels).to(device)
        rand_pts = points * torch.rand((points.shape[0], 1)).to(device)

        kd_tree = KDTree(self.surface)
        rand_sdf = kd_tree.query(rand_pts.detach().cpu().numpy())
        rand_sdf = torch.from_numpy(rand_sdf[0][:, 0])

        samples = []
        for i in range(self.num_latents):
            voxel = voxels[i, :]
            voxel_points = points - voxel
            dist = torch.norm(voxel_points, p=2, dim=-1)
            selector = dist < (1.5 * self.voxel_size)
            voxel_points = voxel_points[selector, :]
            voxel_normals = normals[selector, :]
            voxel_weights = weights[selector]

            if voxel_points.shape[0] > self.max_sample:
                selector = torch.randperm(voxel_points.shape[0])[
                    :self.max_sample]
                voxel_points = voxel_points[selector, :]
                voxel_normals = voxel_normals[selector, :]
                voxel_weights = voxel_weights[selector]

            rand_sample = rand_pts - voxel
            dist = torch.norm(rand_sample, p=2, dim=-1)
            selector = dist < (1.5 * self.voxel_size)
            rand_sample = rand_sample[selector, :]
            rand_weight = weights[selector]
            rand_sdf_sample = rand_sdf[selector]

            rotation = torch.from_numpy(
                self.rotations[i, ...]).float().to(device)
            centroid = torch.from_numpy(
                self.centroids[i, ...]).float().to(device)
            local_pts = torch.rand((256, 3)).to(device)
            local_pts = (local_pts * 3-1.5) * self.voxel_size
            local_sdf = torch.ones((256,)) * -1
            local_weight = torch.zeros((256, ), device=device)
            local_pts = torch.matmul(local_pts, rotation) + centroid

            d1 = torch.randn((voxel_points.shape[0], 1)) * 0.015
            d2 = torch.randn((voxel_points.shape[0], 1)) * 0.01
            sample_pts = torch.cat([
                voxel_points,
                voxel_points + voxel_normals*d1.to(device),
                voxel_points + voxel_normals*d2.to(device),
                rand_sample,
                local_pts
            ], axis=0)
            sample_sdf = torch.cat([
                torch.zeros((voxel_points.shape[0],)),
                torch.zeros((voxel_points.shape[0],))+d1[..., 0],
                torch.zeros((voxel_points.shape[0],))+d2[..., 0],
                rand_sdf_sample,
                local_sdf
            ], axis=0)
            sample_weights = torch.cat([
                voxel_weights,
                voxel_weights,
                voxel_weights,
                rand_weight,
                local_weight
            ], axis=0)

            # print(rotation)
            # self.display_sdf(sample_pts.detach().cpu().numpy(),
            #                  sample_sdf.detach().cpu().numpy())

            sample = torch.zeros((sample_pts.shape[0], 6), device=device)
            sample[:, 0] = i
            sample[:, 1: 4] = sample_pts
            sample[:, 4] = sample_sdf
            sample[:, 5] = sample_weights
            samples.append(sample.detach().cpu().numpy())
        return np.concatenate(samples, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    train_ds = SceneDataset(args.path, training=True)
