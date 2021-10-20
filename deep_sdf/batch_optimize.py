import argparse
import json
import os

import numpy as np
import torch
import trimesh

from dataset import SampleDataset
from network import ImplicitNet
from optimizer import LatentOptimizer
from reconstruct import ShapeReconstructor
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--clamp-dist", type=float, default=-1)
    parser.add_argument("--max-surf-dist", type=float, default=0.05)
    parser.add_argument("--ckpt-freq", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--recon", action='store_true')
    parser.add_argument("--orient", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')

    net_args = json.load(open(args.cfg, 'r'))
    net_params = net_args['params']
    network = ImplicitNet(**net_params).to(device)
    load_model(args.ckpt, network, device)

    splits = json.load(open(args.split, 'r'))
    for scene_name, num_frag in splits.items():
        for i in range(num_frag):
            # if i != 0 and i != 3:
            #     continue
            # if args.skip != 0 and i < args.skip:
            #     continue
            input_dir = os.path.join(args.input, scene_name, str(i))
            output_dir = os.path.join(args.output, scene_name, str(i))
            print('optimising latents {}'.format(input_dir))
            os.makedirs(output_dir, exist_ok=True)
            log_dir = os.path.join(output_dir, "logs")
            eval_data = SampleDataset(
                input_dir, args.orient, args.crop, training=True)
            latent_optim = LatentOptimizer(
                network,
                eval_data.voxels,
                eval_data.samples,
                eval_data.num_latents,
                net_params['latent_dim'],
                eval_data.voxel_size,
                eval_data.centroids,
                eval_data.rotations,
                args.init_lr,
                args.ckpt_freq,
                args.batch_size,
                log_dir=log_dir,
                output=output_dir,
                device=device)

            latent_optim.init_latents()
            latent_optim(args.num_epochs)
            filename = os.path.join(output_dir, "latent_vecs.npy")
            latent_optim.save_latents(filename)

            if args.recon:
                recon = ShapeReconstructor(
                    network,
                    latent_optim.latent_vecs,
                    eval_data.voxels,
                    eval_data.voxel_size,
                    centroids=eval_data.centroids,
                    rotations=eval_data.rotations,
                    surface_pts=eval_data.surface,
                    max_surface_dist=args.max_surf_dist,
                    device=device
                )
                # recon = recon.reconstruct_interp()
                recon = recon.reconstruct()
                if recon is not None:
                    recon.export(os.path.join(output_dir, "recon.ply"))
