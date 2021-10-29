import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def collate_batch(batch):
    index, pnts, sdf = zip(*batch)
    index = torch.from_numpy(np.concatenate(index, axis=0))
    pnts = torch.from_numpy(np.concatenate(pnts, axis=0))
    sdf = torch.from_numpy(np.concatenate(sdf, axis=0))
    perm = torch.randperm(index.shape[0])
    return index[perm], pnts[perm, :], sdf[perm]


class SingleMeshDataset(Dataset):
    def __init__(self, data_path, precompute=False) -> None:
        super().__init__()
        samples = np.load(os.path.join(
            data_path, 'samples.npz'))
        self.samples = samples['samples'].astype(np.float32)
        self.voxels = samples['voxels'].astype(np.float32)
        self.voxel_size = samples['voxel_size']
        self.num_latents = self.voxels.shape[0]
        self.precompute = precompute
        if precompute:
            self.pre_compute()

    def pre_compute(self):
        data = []
        print("starting preparing dataset")
        for i in range(self.num_latents):
            voxel = self.voxels[i, :]
            pnts = self.samples[:, :3] - voxel
            selector = np.linalg.norm(pnts, ord=2, axis=-1)
            selector = selector < (1.5 * self.voxel_size)
            pnts = pnts[selector, :] / (1.5*self.voxel_size)
            sdf = self.samples[selector, 3] / (1.5*self.voxel_size)
            indices = np.asarray([i]*pnts.shape[0])
            data.append(np.concatenate(
                [indices[:, None], pnts, sdf[:, None]], axis=-1))
        print("finished preparing dataset")
        self.samples = np.concatenate(data, axis=0).astype(np.float32)
        np.save('a.npy', self.samples)

    @classmethod
    def get_loader(cls, data_path, batch_size, precompute):
        dataset = SingleMeshDataset(data_path, precompute=precompute)
        return DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=8,
            collate_fn=None if dataset.precompute else collate_batch)

    def __len__(self):
        if self.precompute:
            return self.samples.shape[0]
        else:
            return self.num_latents

    def __getitem__(self, index):
        if self.precompute:
            indices = self.samples[index, 0]
            pnts = self.samples[index, 1:4]
            sdf = self.samples[index, 4]
            return (indices, pnts, sdf)
        else:
            voxel = self.voxels[index, :]
            pnts = self.samples[:, :3] - voxel
            selector = np.linalg.norm(pnts, ord=2, axis=-1)
            selector = selector < (2 * self.voxel_size)
            pnts = pnts[selector, :] / (2*self.voxel_size)
            sdf = self.samples[selector, 3] / (2*self.voxel_size)
            indices = np.asarray([index]*pnts.shape[0])
            return (indices, pnts, sdf)


class BatchMeshDataset(Dataset):
    def __init__(self,
                 data_path: str):
        super().__init__()
        self.data_path = data_path
        cat_dirs = os.listdir(data_path)
        data_points = []
        voxel_count = 0
        print("loading data...")
        start_time = time.time()
        for cat in cat_dirs:
            model_files = os.listdir(os.path.join(data_path, cat))
            cat_models = []
            for ind, filename in enumerate(model_files):
                cat_models.append(filename)
                data_point = np.load(os.path.join(data_path, cat, filename))
                num_voxels = data_point[-1, 0]+1
                data_point[:, 0] += voxel_count
                voxel_count += num_voxels
                data_points.append(data_point)
        self.samples = np.concatenate(data_points, axis=0)
        self.num_latents = voxel_count
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


if __name__ == '__main__':
    dataset = BatchMeshDataset.get_loader(sys.argv[1], 10000)
