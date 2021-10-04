import argparse
import json
import os

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

from dataloader import load_eval_dataset
from network import ImplicitNet


class ShapeReconstructor(object):
    def __init__(self,
                 network,
                 voxels,
                 surface,
                 voxel_size,
                 centroids=None,
                 rotations=None,
                 resolution=8,
                 device=torch.device('cpu')):
        self.device = device
        self.voxels = voxels
        self.surface = surface
        self.network = network
        self.voxel_size = voxel_size
        self.resolution = resolution + 1
        self.latent_vecs = network.latent_vecs
        self.grid_pts, self.xyz = self.get_grid_points()

    def get_grid_points(self, range=[-.5, .5]):
        x = y = z = torch.linspace(range[0], range[1], self.resolution)
        xx, yy, zz = torch.meshgrid(x, y, z)
        grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = grid_points.view(-1, 3)
        xyz = [x, y, z]
        return grid_points, xyz

    def get_sdf(self, latent_vec, grid_pts, batch_size=32**3):
        z = []
        for i, pnts in enumerate(torch.split(grid_pts, batch_size, dim=0)):
            latent = latent_vec.expand(pnts.shape[0], -1)
            grid_sample = torch.cat([latent, pnts], dim=-1)
            z.append(self.network(grid_sample))
        z = torch.cat(z, dim=0)
        return z

    def trace_surface_points(self, z, mask):
        if not isinstance(z, np.ndarray):
            z = z.detach().cpu().numpy()

        spacing = (self.xyz[0][2] - self.xyz[0][1],
                   self.xyz[0][2] - self.xyz[0][1],
                   self.xyz[0][2] - self.xyz[0][1])
        try:
            return marching_cubes(
                volume=z, level=0,
                spacing=spacing, mask=mask)
        except:
            return None

    def reconstruct(self):
        mesh_verts = []
        mesh_faces = []
        num_exist_verts = 0
        self.network.eval()
        for ind in range(self.voxels.shape[0]):
            z_mask = None
            grid_pts = self.grid_pts
            voxel = self.voxels[ind, ...]
            latent_vec = self.latent_vecs[ind, :]
            z = self.get_sdf(latent_vec, grid_pts)
            z = z.detach().cpu().numpy()
            has_surface = np.min(z) < 0 and np.max(z) > 0
            if has_surface:
                z = z.reshape(self.resolution,
                              self.resolution,
                              self.resolution)
                surface = self.trace_surface_points(z, z_mask)
                verts, faces, _, _ = surface
                verts -= .5
                verts *= self.voxel_size
                verts += voxel

                faces += num_exist_verts
                mesh_verts.append(verts)
                mesh_faces.append(faces)
                num_exist_verts += verts.shape[0]

        if (len(mesh_verts) == 0):
            print("reconstruction failed.")
            return None
        else:
            mesh_verts = np.concatenate(mesh_verts, axis=0)
            mesh_faces = np.concatenate(mesh_faces, axis=0)
            recon_shape = trimesh.Trimesh(mesh_verts, mesh_faces)
            return recon_shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')

    voxels, surface, voxel_size, centroids, rotations = load_eval_dataset(
        args.data, load_surface=args.crop, load_transform=args.orient)

    decoder = ImplicitNet.load_from_checkpoint(
        checkpoint_path=args.ckpt)
    decoder.eval()
    print(decoder.latent_vecs)
    recon = ShapeReconstructor(
        decoder, voxels,
        surface, voxel_size, centroids, rotations,
        resolution=8, device=torch.device('cpu'))
    recon_shape = recon.reconstruct()

    if args.output is not None:
        recon_shape.export(args.output)
