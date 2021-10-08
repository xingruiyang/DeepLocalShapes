import pickle
import os
import argparse
import numpy as np
import torch
import trimesh

import mesh_to_sdf
from transformer import PointNetTransformer
from utils import load_model


class MeshSampler(object):
    def __init__(self,
                 voxel_size,
                 pts_per_voxel=4096,
                 network=None,
                 normalize=False,
                 use_depth=False):
        super(MeshSampler, self).__init__()
        self.voxel_size = voxel_size
        self.pts_per_voxel = pts_per_voxel
        self.network = network
        if network is not None:
            self.network = PointNetTransformer()
            load_model(network, self.network)
            self.network.eval()
        self.normalize = normalize
        self.use_depth = use_depth

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def get_orientation(self, pts):
        volumes = pts / (1.5 * self.voxel_size)
        volumes = torch.from_numpy(volumes)[None, ...].float()
        rotation = self.network(volumes, transpose_input=True)
        return rotation.squeeze().detach().numpy()

    def sample_sdf(self, mesh, return_surface=False):
        if self.normalize:
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
        surface_points = mesh_to_sdf.get_surface_point_cloud(
            mesh, scan_count=100)

        voxels = surface_points.points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= self.voxel_size
        print("{} voxels to be sampled".format(voxels.shape[0]))

        samples = []
        centroids = []
        rotations = []

        surface = []
        rand_surface = []
        rand_sdf = []

        num_aligned = 0

        for vid in range(voxels.shape[0]):
            print("{}/{}".format(vid, voxels.shape[0]))
            voxel = voxels[vid, :]
            pcd = surface_points.points - voxel
            dist = np.linalg.norm(pcd, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            pcd = pcd[selector, :]

            if pcd.shape[0] > self.pts_per_voxel:
                pcd = pcd[np.random.permutation(
                    pcd.shape[0])[:self.pts_per_voxel], :]

            centroid = np.mean(pcd, axis=0)
            rotation = np.eye(3)
            if self.network is not None and pcd.shape[0] == self.pts_per_voxel:
                rotation = self.get_orientation(pcd-centroid)
                num_aligned += 1

            query_points = []
            query_points.append(pcd+np.random.randn(*pcd.shape)*0.05)
            query_points.append(pcd+np.random.randn(*pcd.shape)*0.005)
            query_points.append(
                (np.random.rand(512, 3)*3-1.5)*self.voxel_size)
            query_points = np.concatenate(query_points, axis=0)

            sdf, mask = surface_points.get_sdf(
                query_points + voxel,
                use_depth_buffer=self.use_depth)
            sdf = sdf[mask]
            query_points = query_points[mask]

            surface.append(pcd+voxel)
            rand_surface.append(query_points+voxel)
            rand_sdf.append(sdf)

            vsample = np.zeros((query_points.shape[0], 6))
            vsample[:, 0] = float(vid)
            vsample[:, 1:4] = query_points
            vsample[:, 4] = sdf
            vsample[:, 5] = 1
            samples.append(vsample)
            centroids.append(centroid)
            rotations.append(rotation)

        samples = np.concatenate(samples, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)

        print("{} points sampled with {} voxels aligned".format(
            samples.shape[0], rotations.shape[0]))

        surface = np.concatenate(surface, axis=0)
        rand_surface = np.concatenate(rand_surface, axis=0)
        rand_sdf = np.concatenate(rand_sdf, axis=0)

        if return_surface:
            return samples, voxels, centroids, rotations, \
                surface, rand_surface, rand_sdf
        else:
            return samples, voxels, centroids, rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--pts-per-voxel', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--use-depth', action='store_true')
    parser.add_argument('--random-rot', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    sampler = MeshSampler(
        args.voxel_size,
        args.pts_per_voxel,
        args.network,
        args.normalize,
        args.use_depth)

    mesh = trimesh.load(args.mesh)

    samples = sampler.sample_sdf(mesh, return_surface=True)
    samples, voxels, centroids, rotations, \
        surface, rand_surface, rand_sdf = samples
    surface_sdf = np.concatenate(
        [rand_surface, rand_sdf[:, None]], axis=-1)

    sample_name = 'samples.npy'
    surface_pts_name = 'surface_pts.npy'
    surface_sdf_name = 'surface_sdf.npy'

    out = dict()
    out['samples'] = sample_name
    out['surface_pts'] = surface_pts_name
    out['surface_sdf'] = surface_sdf_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
    np.save(os.path.join(args.output, surface_pts_name),
            surface.astype(np.float32))
    np.save(os.path.join(args.output, surface_sdf_name),
            surface_sdf.astype(np.float32))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(True)
    mesh.export(os.path.join(args.output, "gt.ply"))
