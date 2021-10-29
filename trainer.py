import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from datasets import BatchMeshDataset, SingleMeshDataset
from network import ImplicitNet


class Trainer():
    def __init__(
            self,
            out_path,
            device,
            ckpt_freq,
            num_epochs,
            batch_size) -> None:
        self.out_path = out_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.ckpt_freq = ckpt_freq

    def fit(self, model, train_data, centroids=None, rotations=None):
        model.train()
        num_points = train_data.shape[0]
        num_batch = (num_points-1) // self.batch_size + 1
        optimizer = model.configure_optimizers()

        if centroids is not None:
            centroids = torch.from_numpy(centroids).to(self.device)
        if rotations is not None:
            rotations = torch.from_numpy(rotations).to(self.device)

        for n_epoch in range(self.num_epochs):
            print("shuffling inbetween batches")
            start_time = time.time()
            np.random.shuffle(train_data)
            print("shuffling took {} seconds to finish".format(
                time.time()-start_time))
            batch_loss = 0
            pbar = tqdm(range(num_batch))
            for n_batch in pbar:
                bstart = n_batch * self.batch_size
                bend = min(num_points, (n_batch+1)*self.batch_size)
                batch_data = train_data[bstart: bend, :]

                batch_data = torch.from_numpy(batch_data)
                batch_data = batch_data.to(self.device)
                latent_ind = batch_data[:, 0].int()
                sample_pnts = batch_data[:, 1:4].float()
                sample_sdf = batch_data[:, 4].float()

                if centroids is not None:
                    centers = torch.index_select(centroids, 0, latent_ind)
                    sample_pnts -= centers

                if rotations is not None:
                    rotation = torch.index_select(rotations, 0, latent_ind)
                    sample_pnts = torch.matmul(
                        sample_pnts.unsqueeze(1), rotation.transpose(-1, -2))
                    sample_pnts = sample_pnts.squeeze()

                losses = model.training_step(
                    (latent_ind, sample_pnts, sample_sdf))
                loss = losses['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                pbar.set_description(
                    "Epoch {} loss: {:.4f}".format(
                        n_epoch, batch_loss/(n_batch+1)
                    )
                )

            if (n_epoch > 0 and self.ckpt_freq > 0):
                if n_epoch % self.ckpt_freq == 0:
                    self.save_ckpt(model, n_epoch)
            self.save_latest(model)

    def save_ckpt(self, model, n_epoch=0):
        model.save_ckpt(os.path.join(
            self.out_path, 'ckpt_e{}_model.pth'.format(n_epoch)))
        model.save_latents(os.path.join(
            self.out_path, 'ckpt_e{}_latents.pth'.format(n_epoch)))

    def save_latest(self, model):
        model.save_ckpt(os.path.join(self.out_path, 'latest_model.pth'))
        model.save_latents(os.path.join(self.out_path, 'latest_latents.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('data')
    parser.add_argument('out_path')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=-1)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    train_data = BatchMeshDataset(args.data, transform=args.orient)
    model = ImplicitNet.create_from_cfg(
        args.cfg, ckpt=args.ckpt, device=device)
    model.initialize_latents(
        train_data.num_latents, device=device)

    trainer = Trainer(
        device=device,
        ckpt_freq=args.ckpt_freq,
        out_path=args.out_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    trainer.fit(
        model,
        train_data.samples,
        train_data.centroids,
        train_data.rotations)

    # model.save_model(os.path.join(args.output, 'last_ckpt.pth'))
    # model.save_latents(os.path.join(args.output, 'last_latents.npy'))
