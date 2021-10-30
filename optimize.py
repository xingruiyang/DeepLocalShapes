import argparse
import os

import torch

from datasets import BatchMeshDataset
from network import ImplicitNet
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('data')
    parser.add_argument('ckpt')
    parser.add_argument('out_path')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=-1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cpu' if args.cpu else 'cuda')
    train_data = BatchMeshDataset(args.data, transform=args.orient)
    model = ImplicitNet.create_from_cfg(
        args.cfg, ckpt=args.ckpt, device=device)
    model.initialize_latents(
        train_data.num_latents, device=device)
    model.freeze_decoder = True

    trainer = Trainer(
        device=device,
        ckpt_freq=args.ckpt_freq,
        out_path=args.out_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
    trainer.fit(
        model,
        train_data.samples,
        train_data.centroids,
        train_data.rotations,
        train_data.latent_map
    )
    
    model.save_latents(os.path.join(args.out_path, 'last_latents.npy'))
    model.save_latents_splits(args.out_path, train_data.latent_map)
