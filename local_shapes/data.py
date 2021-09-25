import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 orient: bool = False,
                 training: bool = True):
        super(SampleDataset, self).__init__()
        self.root_path = data_path
        self.training = training

        raw_data = self.load_pickle('samples.pkl')
        self.voxels = raw_data['voxels']
        self.voxel_size = raw_data['voxel_size']

        self.num_latents = self.voxels.shape[0]

        if training:
            train_data = os.path.join(data_path, raw_data['samples'])
            self.samples = np.load(train_data)
        else:
            self.samples = None

        if orient:
            self.rotations = raw_data.get('rotations', None)
            self.centroids = raw_data.get('centroids', None)
        else:
            self.rotations = None
            self.centroids = None

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
            return indices, points, sdf
        else:
            return index, self.voxels[index, ...].astype(np.float32)
