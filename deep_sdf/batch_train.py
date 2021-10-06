import argparse

from trainer import NetworkTrainer
import json
import argparse
import os

import torch
import trimesh
from dataset import SampleDataset
from network import ImplicitNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--latent-size", type=int, default=125)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--clamp-dist", type=float, default=-1)
    parser.add_argument("--ckpt-freq", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--orient", action='store_true')
    args = parser.parse_args()
    net_args = json.load(open(args.cfg, 'r'))
    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    splits = json.load(open(args.split, 'r'))['ShapeNetV2']
    for key, value in splits.items():
        for shape in value:
            print('training {}'.format(shape))
            input_dir = os.path.join(args.input, key, shape)
            output_dir = os.path.join(args.output, key, shape)
            if os.path.exists(output_dir):
                continue
            os.makedirs(output_dir, exist_ok=True)

            log_dir = os.path.join(output_dir, "logs")
            gt_mesh = trimesh.load(os.path.join(input_dir, 'gt.ply'))

            network = ImplicitNet(**net_args['params']).to(device)

            dataset = SampleDataset(input_dir, args.orient, training=True)
            latent_vecs = torch.zeros((dataset.num_latents, args.latent_size))
            latent_vecs = latent_vecs.to(device)
            torch.nn.init.normal_(latent_vecs, 0, 0.01**2)
            latent_vecs.requires_grad_()

            trainer = NetworkTrainer(
                network,
                latent_vecs,
                dataset.voxels,
                dataset.samples,
                dataset.voxel_size,
                ckpt_freq=args.ckpt_freq,
                output=output_dir,
                log_dir=log_dir,
                init_lr=args.init_lr,
                batch_size=args.batch_size,
                clamp_dist=args.clamp_dist,
                centroids=dataset.centroids,
                rotations=dataset.rotations,
                gt_mesh=gt_mesh,
                device=device)
            trainer.train(args.num_epochs)
