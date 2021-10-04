import os
import pickle

import numpy as np
from torch.utils.data import DataLoader, Dataset


class ShapeDataset(Dataset):
    def __init__(self,
                 path,
                 training=False,
                 load_samples=False,
                 load_surface=False,
                 load_transform=False,
                 **kwargs):
        super(ShapeDataset, self).__init__(**kwargs)

        raw_data = pickle.load(
            open(os.path.join(path, 'samples.pkl'), "rb"))
        self.voxels = raw_data['voxels'].astype(np.float32)
        self.voxel_size = raw_data['voxel_size']
        self.num_latents = self.voxels.shape[0]
        self.samples = None
        self.surface = None
        self.rotations = None
        self.centroids = None
        self.training = training

        if load_samples:
            train_data = os.path.join(path, raw_data['samples'])
            self.samples = np.load(train_data).astype(np.float32)
            self.num_samples = self.samples.shape[0]
        if load_surface:
            eval_data = os.path.join(path, raw_data['surface'])
            self.surface = np.load(eval_data).astype(np.float32)
        if load_transform:
            self.rotations = raw_data.get('rotations', None)
            self.centroids = raw_data.get('centroids', None)

    def __len__(self):
        if self.training:
            return self.num_samples
        else:
            return self.voxels.shape[0]

    def __getitem__(self, index):
        if self.training:
            indices = self.samples[index, 0].astype(int)
            points = self.samples[index, 1:4].astype(np.float32)
            sdf = self.samples[index, 4].astype(np.float32)
            weights = self.samples[index, 5].astype(np.float32)
            if self.centroids is None:
                return indices, points, sdf, weights
            else:
                centroid = self.centroids[index, ...]
                rotation = self.rotations[index, ...]
                return indices, points, sdf, weights, centroid, rotation
        else:
            if self.centroids is None:
                return self.voxels[index, ...]
            else:
                centroid = self.centroids[index, ...]
                rotation = self.rotations[index, ...]
                return self.voxels[index, ...], centroid, rotation


def load_train_dataset(
        path, load_samples=False, load_surface=False,
        load_transform=False, batch_size=None, shuffle=False, num_workers=8):
    dataset = ShapeDataset(
        path, True, load_samples, load_surface, load_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), \
        dataset.num_latents, dataset.voxel_size


def load_eval_dataset(
        path, load_surface=False, load_transform=False):
    dataset = ShapeDataset(
        path, training=False, load_samples=False,
        load_surface=load_surface, load_transform=load_transform
    )
    return dataset.voxels, dataset.surface, dataset.voxel_size,\
        dataset.centroids, dataset.rotations