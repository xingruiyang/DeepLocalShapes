import json
import argparse
import copy

import numpy as np
import open3d as o3d
import torch
import trimesh
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors

from dataset import SampleDataset
from network import ImplicitNet
from utils import load_latents, load_model


class ShapeReconstructor(object):
    def __init__(self,
                 network,
                 latent_vecs,
                 voxels,
                 voxel_size,
                 resolution=8,
                 centroids=None,
                 rotations=None,
                 surface_pts=None,
                 max_surface_dist=0.1,
                 device=torch.device('cpu')):
        super(ShapeReconstructor, self).__init__()
        self.latent_vecs = latent_vecs
        self.voxels = voxels
        self.network = network
        self.voxel_size = voxel_size
        self.device = device
        self.resolution = resolution+1
        self.max_surface_dist = max_surface_dist
        self.surface_pts = surface_pts
        if surface_pts is not None:
            self.surface_pts = NearestNeighbors(
                n_neighbors=1, metric='l2').fit(surface_pts)
        if isinstance(centroids, np.ndarray):
            self.centroids = torch.from_numpy(
                centroids).to(device).float()
        else:
            self.centroids = centroids

        if isinstance(rotations, np.ndarray):
            self.rotations = torch.from_numpy(
                rotations).to(device).float()
        else:
            self.rotations = rotations
        self.grid_pts, self.xyz = self.get_grid_points(
            self.resolution, range=[-.5, .5])
        self.spacing = (
            self.xyz[0][2] - self.xyz[0][1],
            self.xyz[0][2] - self.xyz[0][1],
            self.xyz[0][2] - self.xyz[0][1])

    def get_grid_points(self, resolution, range=[-1, 1]):
        x = y = z = torch.linspace(range[0], range[1], resolution)
        xx, yy, zz = torch.meshgrid(x, y, z)
        grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = grid_points.view(-1, 3).to(self.device)
        xyz = [x, y, z]
        return grid_points, xyz

    def get_sdf(self, network, latent_vec, grid_points, batch_size=32**3):
        z = []
        for i, pnts in enumerate(torch.split(grid_points, batch_size, dim=0)):
            latent = latent_vec.expand(pnts.shape[0], -1)
            grid_sample = torch.cat([latent, pnts], dim=-1)
            z.append(network(grid_sample))
        z = torch.cat(z, dim=0)
        return z

    def trace_surface_points(self, z, mask):
        if not isinstance(z, np.ndarray):
            z = z.detach().cpu().numpy()
        try:
            return marching_cubes(
                volume=z, level=0,
                spacing=self.spacing, mask=mask)
        except:
            return None

    def get_sdf_grid(self, z_array, z_masks, default_sdf_value=1000):
        voxels = self.voxels / self.voxel_size - .5
        min_voxel = np.round(np.amin(voxels, axis=0)).astype(int)
        max_voxel = np.round(np.amax(voxels, axis=0)).astype(int)
        grid_size = max_voxel - min_voxel + 1
        grid_sdf = np.zeros(grid_size * (self.resolution-1) + 1).astype(float)
        grid_mask = np.zeros(grid_size * (self.resolution-1) + 1).astype(bool)
        grid_mask[...] = False
        grid_sdf[...] = default_sdf_value

        for i in range(voxels.shape[0]):
            voxel = np.round(voxels[i, :]).astype(int) - min_voxel
            z = z_array[i].reshape(
                self.resolution, self.resolution, self.resolution)
            z0 = grid_sdf[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                          voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                          voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]]
            z = np.minimum(z0, z)
            grid_sdf[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                     voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                     voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]] = z
            grid_mask[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                      voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                      voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]] = z_masks[i]
        return grid_sdf, grid_mask, min_voxel

    def reconstruct_interp(self):
        self.network.eval()
        z_array = []
        z_masks = []
        for latent_ind in range(self.voxels.shape[0]):
            voxel_grid = copy.deepcopy(self.grid_pts)
            z_mask = np.zeros(
                (self.resolution, self.resolution, self.resolution)).astype(bool)
            z_mask[...] = True
            if self.surface_pts is not None:
                voxel = self.voxels[latent_ind, :]
                z_mask = self.surface_pts.kneighbors(
                    (voxel_grid.detach().cpu().numpy()) * self.voxel_size + voxel)[0]
                z_mask = z_mask.reshape(
                    self.resolution, self.resolution, self.resolution) < self.max_surface_dist

            if self.centroids is not None:
                voxel_grid -= self.centroids[latent_ind, :] / self.voxel_size
            if self.rotations is not None:
                voxel_grid = torch.matmul(
                    voxel_grid, self.rotations[latent_ind, ...].transpose(0, 1))

            latent_vec = self.latent_vecs[latent_ind, :]
            z = self.get_sdf(self.network, latent_vec, voxel_grid)
            z_array.append(z.detach().cpu().numpy())
            z_masks.append(z_mask)

        z_array, z_masks, min_voxel = self.get_sdf_grid(z_array, z_masks)
        if (np.min(z_array) < 0 and np.max(z_array) > 0):
            surface = self.trace_surface_points(z_array, z_masks)
            if surface is not None:
                verts, faces, _, _ = surface
                verts += min_voxel
                verts *= self.voxel_size
                return trimesh.Trimesh(verts, faces)
        print("reconstruction failed.")
        return None

    def reconstruct(self):
        self.network.eval()
        mesh_verts = []
        mesh_faces = []
        num_exist_verts = 0
        for latent_ind in range(self.voxels.shape[0]):
            voxel = self.voxels[latent_ind, :]
            voxel_grid = copy.deepcopy(self.grid_pts)
            z_mask = None
            if self.surface_pts is not None:
                z_mask = self.surface_pts.kneighbors(
                    (voxel_grid.detach().cpu().numpy()) * self.voxel_size+voxel)[0]
                z_mask = z_mask.reshape(
                    self.resolution, self.resolution, self.resolution) < self.max_surface_dist

            if self.centroids is not None:
                centroid = self.centroids[latent_ind, :]
                voxel_grid -= (centroid / self.voxel_size)
            if self.rotations is not None:
                rotation = self.rotations[latent_ind, ...]
                voxel_grid = torch.matmul(voxel_grid, rotation.transpose(0, 1))

            latent_vec = self.latent_vecs[latent_ind, :]
            z = self.get_sdf(self.network, latent_vec, voxel_grid)
            z = z.detach().cpu().numpy()
            if (np.min(z) < 0 and np.max(z) > 0):
                z = z.reshape(self.resolution,
                              self.resolution,
                              self.resolution)
                surface = self.trace_surface_points(z, z_mask)
                if surface is not None:
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
    parser.add_argument('latents', type=str)
    parser.add_argument('network', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--latent-size', type=int, default=125)
    parser.add_argument('--max-surf-dist', type=float, default=0.01)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--show-surf', action='store_true')
    parser.add_argument('--no-interp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    net_args = json.load(open(args.cfg, 'r'))
    network = ImplicitNet(**net_args['params']).to(device)
    load_model(args.network, network, device)
    latent_vecs = load_latents(args.latents, device)

    dataset = SampleDataset(
        args.data, args.orient, args.crop, training=False)
    reconstructor = ShapeReconstructor(
        network, latent_vecs, dataset.voxels,
        dataset.voxel_size, args.resolution,
        dataset.centroids, dataset.rotations,
        dataset.surface, args.max_surf_dist,
        device=device)

    recon_shape = reconstructor.reconstruct(
    ) if args.no_interp else reconstructor.reconstruct_interp()

    if args.output is not None:
        recon_shape.export(args.output)
    else:
        mesh = recon_shape.as_open3d
        mesh.compute_vertex_normals()
        geometry = [mesh]
        if args.show_surf:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dataset.surface)
            geometry += [pcd]
        o3d.visualization.draw_geometries(geometry)
