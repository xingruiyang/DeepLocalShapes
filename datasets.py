import os
import sys
import time

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset


def collate_batch(batch):
    index, pnts, sdf = zip(*batch)
    index = torch.from_numpy(np.concatenate(index, axis=0))
    pnts = torch.from_numpy(np.concatenate(pnts, axis=0))
    sdf = torch.from_numpy(np.concatenate(sdf, axis=0))
    perm = torch.randperm(index.shape[0])
    return index[perm], pnts[perm, :], sdf[perm]


# class SingleMeshDataset(Dataset):
#     def __init__(self, data_path, precompute=False) -> None:
#         super().__init__()
#         samples = np.load(os.path.join(
#             data_path, 'samples.npz'))
#         self.samples = samples['samples'].astype(np.float32)
#         self.voxels = samples['voxels'].astype(np.float32)
#         self.voxel_size = samples['voxel_size']
#         self.num_latents = self.voxels.shape[0]
#         self.precompute = precompute
#         if precompute:
#             self.pre_compute()

#     def pre_compute(self):
#         data = []
#         print("starting preparing dataset")
#         for i in range(self.num_latents):
#             voxel = self.voxels[i, :]
#             pnts = self.samples[:, :3] - voxel
#             selector = np.linalg.norm(pnts, ord=2, axis=-1)
#             selector = selector < (1.5 * self.voxel_size)
#             pnts = pnts[selector, :] / (1.5*self.voxel_size)
#             sdf = self.samples[selector, 3] / (1.5*self.voxel_size)
#             indices = np.asarray([i]*pnts.shape[0])
#             data.append(np.concatenate(
#                 [indices[:, None], pnts, sdf[:, None]], axis=-1))
#         print("finished preparing dataset")
#         self.samples = np.concatenate(data, axis=0).astype(np.float32)
#         np.save('a.npy', self.samples)

#     @classmethod
#     def get_loader(cls, data_path, batch_size, precompute):
#         dataset = SingleMeshDataset(data_path, precompute=precompute)
#         return DataLoader(
#             dataset, batch_size=batch_size,
#             shuffle=True, num_workers=8,
#             collate_fn=None if dataset.precompute else collate_batch)

#     def __len__(self):
#         if self.precompute:
#             return self.samples.shape[0]
#         else:
#             return self.num_latents

#     def __getitem__(self, index):
#         if self.precompute:
#             indices = self.samples[index, 0]
#             pnts = self.samples[index, 1:4]
#             sdf = self.samples[index, 4]
#             return (indices, pnts, sdf)
#         else:
#             voxel = self.voxels[index, :]
#             pnts = self.samples[:, :3] - voxel
#             selector = np.linalg.norm(pnts, ord=2, axis=-1)
#             selector = selector < (2 * self.voxel_size)
#             pnts = pnts[selector, :] / (2*self.voxel_size)
#             sdf = self.samples[selector, 3] / (2*self.voxel_size)
#             indices = np.asarray([index]*pnts.shape[0])
#             return (indices, pnts, sdf)

class BatchMeshDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform: bool = False):
        super().__init__()
        self.data_path = data_path
        cat_dirs = os.listdir(data_path)
        data_points = []
        rotations = []
        centroids = []
        self.latent_map = dict()

        # num_data_points = 0
        # for cat in cat_dirs:
        #     model_files = os.listdir(os.path.join(data_path, cat))
        #     for ind, filename in enumerate(model_files):
        #         cat_filename = os.path.join(cat, filename)
        #         data = np.load(os.path.join(data_path, cat_filename))
        #         num_data_points += data['samples'].shape[0]

        print("loading data...")
        start_time = time.time()
        num_model = 0
        voxel_count = 0
        shape_count =0 
        # self.samples = np.zeros((num_data_points, 5))
        for cat in cat_dirs:
            model_files = os.listdir(os.path.join(data_path, cat))
            cat_models = []
            for ind, filename in enumerate(model_files):
                cat_models.append(filename)
                cat_filename = os.path.join(cat, filename)
                data = np.load(os.path.join(data_path, cat_filename))
                data_point = data['samples']

                num_voxels = int(data_point[-1, 0]+1)
                # self.samples[shape_count:(
                #     shape_count+data_point.shape[0]), :] = data_point
                self.latent_map[cat_filename.split('.')[0]] = (
                    voxel_count, voxel_count+num_voxels)

                data_point[:, 0] += voxel_count
                voxel_count += num_voxels
                # shape_count += num_voxels
                data_points.append(data_point)

                if transform:
                    rotation = data['rotations']
                    centroid = data['centroids']
                    rotations.append(rotation)
                    centroids.append(centroid)
                num_model += 1
        self.samples = np.concatenate(data_points, axis=0)
        self.num_latents = voxel_count
        if transform:
            self.rotations = np.concatenate(rotations, axis=0)
            self.centroids = np.concatenate(centroids, axis=0)
        else:
            self.rotations = None
            self.centroids = None
        print("data loaded for {} seconds".format(time.time()-start_time))

    @classmethod
    def get_loader(cls, data_path, batch_size):
        dataset = BatchMeshDataset(data_path)
        return DataLoader(dataset, batch_size=batch_size,
                          num_workers=8, shuffle=True)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        indices = self.samples[index, 0]
        pnts = self.samples[index, 1:4]
        sdf = self.samples[index, 4]
        return (indices, pnts, sdf)


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
            sample = torch.zeros((sample_pts.shape[0], 6), device=device)
            sample[:, 0] = i
            sample[:, 1: 4] = sample_pts
            sample[:, 4] = sample_sdf
            sample[:, 5] = sample_weights
            samples.append(sample.detach().cpu().numpy())
        return np.concatenate(samples, axis=0)
