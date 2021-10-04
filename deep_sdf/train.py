import argparse
import json
import os

import torch

from dataloader import load_train_dataset
from network import ImplicitNet


class Trainer():
    def __init__(self, network, gpu=True):
        self.network = network
        self.network.train()
        self.device = torch.device('cuda' if gpu else 'cpu')

    def fit(self, network, train_data):
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')

    train_data, num_latents, voxel_size = load_train_dataset(
        args.data, load_samples=True,
        load_transform=args.orient,
        batch_size=args.batch_size,
        shuffle=True)

    net_config = json.load(open(args.cfg))
    net_params = net_config["params"]
    network = ImplicitNet(**net_params, voxel_size=voxel_size)
    network.initialize_latents(num_latents, device)

    trainer = Trainer()
    trainer.fit(network, train_data)
