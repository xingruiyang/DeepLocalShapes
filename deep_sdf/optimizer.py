import json
import argparse
import os

import numpy as np
import torch
import trimesh
from torch.utils.tensorboard import SummaryWriter

from dataset import SampleDataset
from losses import chamfer_distance
from network import ImplicitNet
from reconstruct import ShapeReconstructor
from utils import load_model, log_progress


def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class LatentOptimizer(object):
    def __init__(self,
                 network,
                 voxels,
                 eval_data,
                 num_latents,
                 latent_size,
                 voxel_size,
                 centroids,
                 rotations,
                 init_lr,
                 ckpt_freq=-1,
                 batch_size=512,
                 clamp_dist=0.1,
                 gt_mesh=None,
                 log_dir=None,
                 output=None,
                 device=torch.device('cpu')) -> None:
        self.global_steps = 0
        self.num_samples = 2**14
        self.init_lr = init_lr
        self.voxel_size = voxel_size
        self.device = device
        self.batch_size = batch_size
        self.num_batch = (eval_data.shape[0]-1)//batch_size+1
        self.eval_data = torch.from_numpy(eval_data).to(device)
        self.network = network
        self.logger = SummaryWriter(log_dir) if log_dir is not None else None
        self.output = output
        self.ckpt_freq = ckpt_freq
        self.clamp_dist = clamp_dist
        self.clamp = clamp_dist > 0
        self.gt_points = gt_mesh.sample(
            self.num_samples) if gt_mesh is not None else None
        self.centroids = None
        self.rotations = None
        self.voxels = voxels

        if centroids is not None:
            self.centroids = torch.from_numpy(centroids).float()
            self.centroids = self.centroids.to(device)
        if rotations is not None:
            self.rotations = torch.from_numpy(rotations).float()
            self.rotations = self.rotations.to(device)

        self.latent_vecs = torch.zeros((num_latents, latent_size))
        self.latent_vecs = self.latent_vecs.to(device)
        self.latent_vecs.requires_grad_()
        self.optimizer = torch.optim.Adam(
            [self.latent_vecs], lr=init_lr)

    def init_latents(self):
        torch.nn.init.normal_(self.latent_vecs, 0, 0.01**2)

    def __call__(self, num_epochs):
        self.network.train()
        self.global_steps = 0
        input_scale = 1.0 / self.voxel_size
        for n_iter in range(num_epochs):
            batch_loss = 0
            batch_sdf_loss = 0
            batch_latent_loss = 0
            batch_steps = 0
            eval_data = self.eval_data[torch.randperm(
                self.eval_data.shape[0]), :]
            for batch_idx in range(self.num_batch):
                begin = batch_idx * self.batch_size
                end = min(eval_data.shape[0], (batch_idx+1)*self.batch_size)

                latent_ind = eval_data[begin:end, 0].to(self.device).int()
                sdf_values = eval_data[begin:end, 4].to(self.device) * input_scale
                points = eval_data[begin:end, 1:4].to(self.device) * input_scale
                weights = eval_data[begin:end, 5].to(self.device)
                latents = torch.index_select(self.latent_vecs, 0, latent_ind)

                if self.centroids is not None:
                    centroid = torch.index_select(
                        self.centroids, 0, latent_ind)
                    points -= centroid * input_scale

                if self.rotations is not None:
                    rotation = torch.index_select(
                        self.rotations, 0, latent_ind)
                    points = torch.bmm(
                        points.unsqueeze(1),
                        rotation.transpose(1, 2)).squeeze()

                points = torch.cat([latents, points], dim=-1)
                surface_pred = self.network(points).squeeze()

                if self.network.use_tanh:
                    sdf_values = torch.tanh(sdf_values)

                if self.network.clamp:
                    surface_pred = torch.clamp(
                        surface_pred, -self.network.clamp_dist, self.network.clamp_dist)
                    sdf_values = torch.clamp(
                        sdf_values, -self.network.clamp_dist, self.network.clamp_dist)

                sdf_loss = (((sdf_values-surface_pred)*weights).abs()).mean()
                latent_loss = latents.abs().mean()
                loss = sdf_loss + latent_loss * 1e-3
                # gradient = compute_gradient(surface_pred, points)[
                #     weights == 0, -3:]
                # grad_loss = torch.abs(gradient.norm(dim=-1) - 1).mean()
                # inter_loss = torch.exp(-1e2 * torch.abs(surface_pred[weights==0])).mean()
                # loss = sdf_loss + latent_loss * 1e-3 + 1e-1 * grad_loss  # + 1e-2 * inter_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                batch_sdf_loss += sdf_loss.item()
                batch_latent_loss += latent_loss.item()
                batch_steps += 1
                self.global_steps += 1

                if self.logger is not None:
                    self.logger.add_scalar(
                        'eval/sdf_loss', sdf_loss.item(), self.global_steps)
                    self.logger.add_scalar(
                        'eval/latent_loss', latent_loss.item(), self.global_steps)

                log_progress(
                    batch_steps, self.num_batch,
                    "Optimizing iter {}".format(n_iter),
                    "loss: {:.4f} sdf: {:.4f} latent: {:.4f}".format(
                        batch_loss/batch_steps,
                        batch_sdf_loss/batch_steps,
                        batch_latent_loss/batch_steps))

            self.save_latents(os.path.join(self.output, 'latest_latents.npy'))
            if (self.ckpt_freq > 0 and n_iter % self.ckpt_freq == 0) or n_iter == (num_epochs-1):
                self.save_ckpt(n_iter)

    def update_lr(self, iter, init_lr):
        lr = init_lr if iter == 0 else init_lr * (0.9**(iter//100))
        for _, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        print("learning rate is adjusted to {}".format(lr))

    def save_latents(self, filename):
        latent_vecs = self.latent_vecs.detach().cpu().numpy()
        np.save(filename, latent_vecs)

    def save_ckpt(self, epoch):
        print("saving ckpt for epoch {}".format(epoch))
        self.save_latents(os.path.join(
            self.output, 'ckpt_{}_latents.npy'.format(epoch)))
        shape = None
        if self.logger is not None:
            with torch.no_grad():
                self.network.eval()
                reconstructor = ShapeReconstructor(
                    self.network,
                    self.latent_vecs,
                    self.voxels,
                    self.voxel_size,
                    centroids=self.centroids,
                    rotations=self.rotations,
                    device=self.device)
                shape = reconstructor.reconstruct_interp()
                if self.gt_points is not None and shape is not None:
                    recon_points = shape.sample(self.num_samples)
                    dist = chamfer_distance(
                        self.gt_points, recon_points, direction='bi')
                    self.logger.add_scalar("eval/chamfer_dist", dist, epoch)
                    shape.export(os.path.join(
                        self.output, "ckpt_{}_mesh.ply".format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument("--gt_mesh", type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--latent_size', type=int, default=125)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--clamp_dist', type=float, default=-1)
    parser.add_argument("--ckpt_freq", type=int, default=-1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    gt_mesh = trimesh.load(args.gt_mesh) if args.gt_mesh is not None else None

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    net_args = json.load(open(args.cfg, 'r'))
    network = ImplicitNet(**net_args['params']).to(device)
    load_model(args.ckpt, network, device)

    eval_data = SampleDataset(
        args.data, args.orient, training=True)
    logger = SummaryWriter(os.path.join(args.output, 'logs/'))
    latent_optim = LatentOptimizer(
        network,
        eval_data.voxels,
        eval_data.samples,
        eval_data.num_latents,
        args.latent_size,
        eval_data.voxel_size,
        eval_data.centroids,
        eval_data.rotations,
        args.init_lr,
        args.ckpt_freq,
        args.batch_size,
        args.clamp_dist,
        gt_mesh=gt_mesh,
        logger=logger,
        output=args.output,
        device=device)

    latent_optim.init_latents()
    latent_optim(args.num_epochs)
    filename = os.path.join(args.output, "latent_vecs.npy")
    latent_optim.save_latents(filename)
