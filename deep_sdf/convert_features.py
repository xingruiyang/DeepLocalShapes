import argparse
import pickle

import numpy as np
import torch

def get_feature_vecs(voxels, latents, rotations, centroids):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('latents')
    parser.add_argument('--method', type=str, default='direct')
    args = parser.parse_args()

    data = pickle.load(open(args.data, 'rb'))
    voxels = data['voxels']
    rotations = data['rotations']
    centroids = data['centroids']
    latent_vecs = np.load(args.latents)

    if args.method == 'direct':
