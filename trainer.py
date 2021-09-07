import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import SampleDataset
from network import ImplicitNet
from utils import log_progress, save_checkpoints, save_latest


class NetworkTrainer(object):
    def __init__(self,
                 network,
                 latent_vecs,
                 train_data,
                 voxel_size,
                 ckpt_freq=-1,
                 init_lr=1e-3,
                 output=None,
                 log_dir=None,
                 batch_size=1000,
                 device=torch.device('cpu')):
        super(NetworkTrainer, self).__init__()
        self.network = network
        self.log_dir = log_dir
        self.device = device
        self.output = output
        self.init_lr = init_lr
        self.voxel_size = voxel_size
        self.latent_vecs = latent_vecs
        self.ckpt_freq = ckpt_freq
        self.batch_size = batch_size
        self.num_batch = (train_data.shape[0]-1)//batch_size+1
        self.train_data = torch.from_numpy(train_data).to(device)

        if log_dir is not None:
            self.logger = SummaryWriter(log_dir)

        self.optimizer = Adam(
            [{'params': network.parameters(), 'lr': init_lr},
             {'params': self.latent_vecs, 'lr': init_lr}])

    def train(self, num_epochs):
        global_steps = 0
        input_scale = 1.0 / self.voxel_size
        # input_scale = 1.0
        for n_iter in range(num_epochs):
            batch_loss = 0
            batch_sdf_loss = 0
            batch_latent_loss = 0
            batch_steps = 0
            self.network.train()
            train_data = self.train_data[torch.randperm(
                self.train_data.shape[0]), :]
            for batch_idx in range(self.num_batch):
                begin = batch_idx * self.batch_size
                end = min(train_data.shape[0], (batch_idx+1)*self.batch_size)

                latent_ind = train_data[begin:end, 0].int()
                sdf_values = train_data[begin:end, 4] * input_scale
                points = train_data[begin:end, 1:4] * input_scale

                latents = torch.index_select(self.latent_vecs, 0, latent_ind)
                points = torch.cat([latents, points], dim=-1)

                surface_pred = self.network(points).squeeze()
                surface_pred = torch.clamp(surface_pred, -0.1, 0.1)
                sdf_values = torch.tanh(sdf_values)
                sdf_values = torch.clamp(sdf_values, -0.1, 0.1)

                sdf_loss = ((sdf_values - surface_pred).abs()).mean()
                latent_loss = latents.abs().mean()
                loss = sdf_loss + latent_loss * 1e-4

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                batch_sdf_loss += sdf_loss.item()
                batch_latent_loss += latent_loss.item()
                batch_steps += 1
                global_steps += 1

                if self.log_dir is not None:
                    self.logger.add_scalar(
                        'loss', loss.item(), global_steps)

                log_progress(
                    batch_steps, self.num_batch,
                    "Optimizing iter {}".format(n_iter),
                    "loss: {:.4f} sdf: {:.4f} latent: {:.4f}".format(
                        batch_loss/batch_steps,
                        batch_sdf_loss/batch_steps,
                        batch_latent_loss/batch_steps))

            save_latest(args.output, self.network,
                        self.optimizer, self.latent_vecs, n_iter)

            if self.ckpt_freq > 0 and n_iter % self.ckpt_freq == 0:
                save_checkpoints(args.output, self.network,
                                 self.optimizer, self.latent_vecs, n_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--latent_size", type=int, default=125)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_freq", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    net_args = {"d_in": args.latent_size + 3, "dims": [128, 128, 128]}
    network = ImplicitNet(**net_args).to(device)

    dataset = SampleDataset(args.data, True)
    latent_vecs = torch.zeros((dataset.num_latents, args.latent_size))
    latent_vecs = latent_vecs.to(device)
    torch.nn.init.normal_(latent_vecs, 0, 0.01**2)
    latent_vecs.requires_grad_()

    trainer = NetworkTrainer(network, latent_vecs,
                             dataset.samples, dataset.voxel_size,
                             ckpt_freq=args.ckpt_freq,
                             log_dir=args.log_dir,
                             init_lr=args.init_lr,
                             batch_size=args.batch_size,
                             device=device)
    trainer.train(args.num_epochs)
