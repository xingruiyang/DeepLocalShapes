# import sys  # noqa
# sys.path.insert(0, '/workspace')  # noqa

import argparse
import os

import numpy as np
import torch
import trimesh
from torch.autograd import grad
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from data import SampleDataset
from network import ImplicitNet
from reconstruct import ShapeReconstructor
from utils import chamfer_distance, log_progress, save_ckpts, save_latest


class NetworkTrainer(object):
    def __init__(self,
                 network,
                 latent_vecs,
                 voxels,
                 train_data,
                 voxel_size,
                 ckpt_freq=-1,
                 init_lr=1e-3,
                 output=None,
                 log_dir=None,
                 batch_size=1000,
                 clamp_dist=0.1,
                 centroids=None,
                 orientations=None,
                 gt_mesh=None,
                 device=torch.device('cpu')):
        super(NetworkTrainer, self).__init__()
        self.num_samples = 2**14
        self.network = network
        self.log_dir = log_dir
        self.device = device
        self.output = output
        self.voxels = voxels
        self.init_lr = init_lr
        self.voxel_size = voxel_size
        self.latent_vecs = latent_vecs
        self.ckpt_freq = ckpt_freq
        self.batch_size = batch_size
        self.clamp_dist = clamp_dist
        self.clamp = clamp_dist > 0
        self.gt_points = gt_mesh.sample(
            self.num_samples) if gt_mesh is not None else None
        self.centroids = None
        self.orientations = None

        if centroids is not None:
            self.centroids = torch.from_numpy(centroids).float()
            self.centroids = self.centroids.to(device)
        if orientations is not None:
            self.orientations = torch.from_numpy(orientations).float()
            self.orientations = self.orientations.to(device)

        self.num_batch = (train_data.shape[0]-1)//batch_size+1
        self.train_data = torch.from_numpy(train_data).float()
        self.global_step = 0

        if log_dir is not None:
            self.logger = SummaryWriter(log_dir)

        self.optimizer = Adam(
            [{'params': network.parameters(), 'lr': init_lr},
             {'params': self.latent_vecs, 'lr': init_lr}])

    def train(self, num_epochs):
        self.global_steps = 0
        input_scale = 1.0 / self.voxel_size
        for n_iter in range(num_epochs):
            self.network.train()
            batch_loss = 0
            batch_sdf_loss = 0
            batch_latent_loss = 0
            batch_steps = 0
            train_data = self.train_data[torch.randperm(
                self.train_data.shape[0]), :]
            for batch_idx in range(self.num_batch):
                begin = batch_idx * self.batch_size
                end = min(train_data.shape[0], (batch_idx+1)*self.batch_size)

                latent_ind = train_data[begin:end, 0].to(device).int()
                sdf_values = train_data[begin:end, 4].to(device) * input_scale
                points = train_data[begin:end, 1:4].to(device)
                weights = train_data[begin:end, 5].to(device)
                latents = torch.index_select(self.latent_vecs, 0, latent_ind)

                if self.centroids is not None:
                    centre = torch.index_select(self.centroids, 0, latent_ind)
                    points -= centre

                # if self.orientations is not None:
                #     orient = torch.index_select(
                #         self.orientations, 0, latent_ind)
                #     points = torch.matmul(
                #         points[:, None, :], orient.transpose(1, 2))

                points *= input_scale
                points = torch.cat([latents, points.squeeze()], dim=-1)

                surface_pred = self.network(points).squeeze()
                sdf_values = torch.tanh(sdf_values)

                if self.clamp:
                    surface_pred = torch.clamp(
                        surface_pred, -self.clamp_dist, self.clamp_dist)
                    sdf_values = torch.clamp(
                        sdf_values, -self.clamp_dist, self.clamp_dist)

                sdf_loss = (((sdf_values-surface_pred)*weights).abs()).mean()
                latent_loss = latents.abs().mean()
                loss = sdf_loss + latent_loss * 1e-3

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                batch_sdf_loss += sdf_loss.item()
                batch_latent_loss += latent_loss.item()
                batch_steps += 1
                self.global_steps += 1

                if self.log_dir is not None:
                    self.logger.add_scalar(
                        'train/sdf_loss', sdf_loss.item(), self.global_steps)
                    self.logger.add_scalar(
                        'train/latent_loss', latent_loss.item(), self.global_steps)

                log_progress(
                    batch_steps, self.num_batch,
                    "Optimizing iter {}".format(n_iter),
                    "loss: {:.4f} sdf: {:.4f} latent: {:.4f}".format(
                        batch_loss/batch_steps,
                        batch_sdf_loss/batch_steps,
                        batch_latent_loss/batch_steps))

            save_latest(self.output, self.network,
                        self.optimizer, self.latent_vecs, n_iter)

            if self.ckpt_freq > 0 and n_iter % self.ckpt_freq == 0:
                self.save_ckpt(n_iter)

    def save_ckpt(self, epoch=0):
        print("saving ckpt for epoch {}".format(epoch))
        save_ckpts(self.output, self.network,
                   self.optimizer,
                   self.latent_vecs, epoch)
        if self.logger is not None:
            with torch.no_grad():
                self.network.eval()
                reconstructor = ShapeReconstructor(
                    self.network,
                    self.latent_vecs,
                    self.voxels,
                    self.voxel_size,
                    centroids=self.centroids,
                    orientations=self.orientations,
                    device=self.device)
                shape, verts, faces = reconstructor.reconstruct_interp(True)
                vertex_tensor = torch.from_numpy(verts).float()
                face_tensor = torch.from_numpy(faces).int()
                self.logger.add_mesh(
                    "train",
                    vertices=vertex_tensor.unsqueeze(0),
                    faces=face_tensor.unsqueeze(0),
                    global_step=epoch)
                if self.gt_points is not None:
                    recon_points = shape.sample(self.num_samples)
                    dist = chamfer_distance(self.gt_points, recon_points)
                    self.logger.add_scalar("train/chamfer_dist", dist, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--gt_mesh", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--latent_size", type=int, default=125)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--clamp_dist", type=float, default=-1)
    parser.add_argument("--ckpt_freq", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--orient", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    log_dir = args.log_dir if args.log_dir is not None else os.path.join(
        args.output, "logs")
    gt_mesh = trimesh.load(args.gt_mesh) if args.gt_mesh is not None else None

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    net_args = {"d_in": args.latent_size + 3, "dims": [128, 128, 128]}
    network = ImplicitNet(**net_args).to(device)

    dataset = SampleDataset(args.data, args.orient, True)
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
        output=args.output,
        log_dir=log_dir,
        init_lr=args.init_lr,
        batch_size=args.batch_size,
        clamp_dist=args.clamp_dist,
        centroids=dataset.centroids,
        orientations=dataset.rotations,
        gt_mesh=gt_mesh,
        device=device)
    trainer.train(args.num_epochs)
