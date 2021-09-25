import sys  # noqa

sys.path.insert(0, '/workspace')  # noqa

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import trimesh
from orientation.transformer import PointNetTransformer
from utils.io_utils import load_model

import mesh_to_sdf


class MeshSampler():
    def __init__(self,
                 mesh,
                 voxel_size,
                 min_surface_pts=2048,
                 transformer=None,
                 normalize=False):
        self.mesh = mesh
        self.voxel_size = voxel_size
        self.min_surface_pts = min_surface_pts
        self.transformer = transformer
        if transformer is not None:
            self.transformer = PointNetTransformer()
            load_model(transformer, self.transformer)
            self.transformer.eval()
        if normalize:
            self.mesh = self.normalize_mesh(mesh)

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def normalize_mesh(self, mesh):
        return mesh_to_sdf.scale_to_unit_cube(mesh)

    def get_centroid_and_orientation(self, pts):
        random_ind = np.random.permutation(
            pts.shape[0])[:self.min_surface_pts]
        voxel_surface = pts[random_ind, :]
        centre = np.mean(voxel_surface, axis=0)
        volume_surface = (voxel_surface-centre) / (1.5*self.voxel_size)
        volume_surface = torch.from_numpy(volume_surface)
        volume_surface = volume_surface[None, ...].float()
        orientation = self.transformer(
            volume_surface, transpose_input=True)
        orientation = orientation.detach().numpy()
        return centre, orientation[0, ...]

    def sample_sdf(self):
        mesh_centre = np.mean(self.mesh.vertices, axis=0)
        bounding_radius = np.max(np.linalg.norm(
            self.mesh.vertices-mesh_centre, axis=1)) * 1.2
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(self.mesh)
        points, sdf, surface_points = surface_point_cloud.sample_sdf_near_surface(
            args.num_samples, sign_method='normal',
            radius=bounding_radius, centre=mesh_centre)

        self.display_sdf(points, sdf)
        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        centroids = []
        rotations = []
        num_oriented_voxels = 0
        for vid in range(voxels.shape[0]):
            voxel_pts = points - voxels[vid, :]
            dist = np.linalg.norm(voxel_pts, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            voxel_pts = voxel_pts[selector, :]
            voxel_sdf = sdf[selector]

            voxel_surface = surface_points - voxels[vid, :]
            dist = np.linalg.norm(voxel_surface, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            voxel_surface = voxel_surface[selector, :]

            centre = np.zeros((3, 1))
            orientation = np.eye(3)
            if self.transformer is not None:
                if voxel_surface.shape[0] >= self.min_surface_pts:
                    centre, orientation = self.get_centroid_and_orientation(
                        voxel_surface)
                    num_oriented_voxels += 1
                else:
                    centre = np.mean(voxel_surface, axis=0)
            centroids.append(centre)
            rotations.append(orientation)

            # self.display_sdf(voxel_pts, voxel_sdf)

            vsample = np.zeros((voxel_pts.shape[0], 6))
            vsample[:, 0] = float(vid)
            vsample[:, 1:4] = voxel_pts
            vsample[:, 4] = voxel_sdf
            vsample[:, 5] = 1
            samples.append(vsample)

        print("total of {} voxels sampled with {} oriented".format(
            voxels.shape[0], num_oriented_voxels))
        samples = np.concatenate(samples, axis=0)
        return samples, voxels, centroids, rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--min_surface_pts', type=int, default=2048)
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--transformer', type=str, default=None)
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    mesh = trimesh.load(args.input)
    sampler = MeshSampler(
        mesh, args.voxel_size,
        args.min_surface_pts,
        args.transformer,
        args.normalize)

    samples, voxels, centroids, rotations = sampler.sample_sdf()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    sample_name = 'samples.npy'
    out['samples'] = sample_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = np.stack(centroids, axis=0).astype(np.float32)
    out['rotations'] = np.stack(rotations, axis=0).astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
