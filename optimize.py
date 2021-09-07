import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SampleDataset
from network import ImplicitNet
from utils import load_model, log_progress


class LatentOptimizer(object):
    def __init__(self,
                 network,
                 eval_data,
                 num_latents,
                 latent_size,
                 voxel_size,
                 init_lr,
                 batch_size=512,
                 logger=None,
                 device=torch.device('cpu')) -> None:

        self.init_lr = init_lr
        self.voxel_size = voxel_size
        self.device = device
        self.batch_size = batch_size
        self.num_batch = (eval_data.shape[0]-1)//batch_size+1
        self.eval_data = torch.from_numpy(eval_data).to(device)
        self.network = network
        self.logger = logger

        self.latent_vecs = torch.zeros((num_latents, latent_size))
        self.latent_vecs = self.latent_vecs.to(device)
        self.latent_vecs.requires_grad_()
        self.optimizer = torch.optim.Adam(
            [self.latent_vecs], lr=init_lr)

    def init_latents(self):
        torch.nn.init.normal_(self.latent_vecs, 0, 0.01**2)

    def __call__(self, num_epochs):
        self.network.eval()
        global_steps = 0
        input_scale = 1.0 / self.voxel_size
        for n_iter in range(num_epochs):
            batch_loss = 0
            batch_sdf_loss = 0
            batch_latent_loss = 0
            batch_steps = 0
            self.network.train()
            eval_data = self.eval_data[torch.randperm(
                self.eval_data.shape[0]), :]
            for batch_idx in range(self.num_batch):
                begin = batch_idx * self.batch_size
                end = min(eval_data.shape[0], (batch_idx+1)*self.batch_size)

                latent_ind = eval_data[begin:end, 0].int()
                sdf_values = eval_data[begin:end, 4] * input_scale
                points = eval_data[begin:end, 1:4] * input_scale

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

                if self.logger is not None:
                    self.logger.add_scalar(
                        'val/loss', loss.item(), global_steps)

                log_progress(
                    batch_steps, self.num_batch,
                    "Optimizing iter {}".format(n_iter),
                    "loss: {:.4f} sdf: {:.4f} latent: {:.4f}".format(
                        batch_loss/batch_steps,
                        batch_sdf_loss/batch_steps,
                        batch_latent_loss/batch_steps))

    def update_lr(self, iter, init_lr):
        lr = init_lr if iter == 0 else init_lr * (0.9**(iter//100))
        for _, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        print("learning rate is adjusted to {}".format(lr))

    def save_latents(self, output):
        latent_vecs = self.latent_vecs.detach().cpu().numpy()
        filename = os.path.join(output, "latent_vecs.npy")
        np.save(filename, latent_vecs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--latent_size', type=int, default=125)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        not args.cpu and torch.cuda.is_available()) else 'cpu')
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    net_args = {"d_in": args.latent_size + 3, "dims": [128, 128, 128]}
    network = ImplicitNet(**net_args).to(device)
    load_model(args.ckpt, network, device)

    eval_data = SampleDataset(args.data, True)
    latent_optim = LatentOptimizer(
        network, eval_data.samples, eval_data.num_latents, args.latent_size,
        eval_data.voxel_size, args.init_lr, args.batch_size, device=device)

    latent_optim.init_latents()
    latent_optim(args.num_iter)
    latent_optim.save_latents(args.output)