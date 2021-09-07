import argparse
import os
import pickle

import numpy as np
import torch
import trimesh

from mesh_to_sdf import mesh_to_sdf


class MeshSampler():
    def __init__(self, mesh, voxel_size):
        self.mesh = mesh
        self.voxel_size = voxel_size

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def sample_sdf(self):
        mesh_centre = np.mean(self.mesh.vertices, axis=0)
        bounding_radius = np.max(np.linalg.norm(
            self.mesh.vertices-mesh_centre, axis=1)) * 1.2
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(self.mesh)
        points, sdf = surface_point_cloud.sample_sdf_near_surface(
            args.num_samples, sign_method='normal', 
            radius=bounding_radius, centre=mesh_centre)

        self.display_sdf(points, sdf)
        surface_points = self.mesh.sample(2**16)
        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        for vid in range(voxels.shape[0]):
            voxel_pts = points - voxels[vid, :]
            dist = np.linalg.norm(voxel_pts, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            voxel_pts = voxel_pts[selector, :]
            voxel_sdf = sdf[selector]
            vsample = np.zeros((voxel_pts.shape[0], 5))
            vsample[:, 0] = float(vid)
            vsample[:, 1:4] = voxel_pts
            vsample[:, 4] = voxel_sdf
            vsample = vsample.astype(np.float32)
            samples.append(vsample)
        samples = np.concatenate(samples, axis=0)
        return samples, voxels


class DepthSampler():
    def __init__(self, mesh, voxel_size):
        self.mesh = mesh
        self.voxel_size = voxel_size

    def _sample_sdf(self):
        mesh_centre = np.mean(self.mesh.vertices, axis=0)
        bounding_radius = np.max(np.linalg.norm(
            self.mesh.vertices-mesh_centre, axis=1)) * 1.2
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(self.mesh)
        points, sdf = surface_point_cloud.sample_sdf_near_surface(
            args.num_samples, sign_method='normal', 
            radius=bounding_radius, centre=mesh_centre)

        self.display_sdf(points, sdf)
        surface_points = self.mesh.sample(2**16)
        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        for vid in range(voxels.shape[0]):
            voxel_pts = points - voxels[vid, :]
            dist = np.linalg.norm(voxel_pts, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            voxel_pts = voxel_pts[selector, :]
            voxel_sdf = sdf[selector]
            vsample = np.zeros((voxel_pts.shape[0], 5))
            vsample[:, 0] = float(vid)
            vsample[:, 1:4] = voxel_pts
            vsample[:, 4] = voxel_sdf
            vsample = vsample.astype(np.float32)
            samples.append(vsample)
        samples = np.concatenate(samples, axis=0)
        return samples, voxels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=1000000)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        print("Using mesh sampler")
        mesh = trimesh.load(args.input)
        sampler = MeshSampler(mesh, args.voxel_size)
    else:
        sampler = DepthSampler(args.input, args.voxel_size)

    samples, voxels = sampler.sample_sdf()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    out['voxels'] = voxels
    out['samples'] = samples
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "sample.pkl"), "wb") as f:
        pickle.dump(out, f)
