from genericpath import isfile
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
                 voxel_size,
                 pts_per_shapes,
                 min_surface_pts=2048,
                 network=None,
                 device=torch.device('cpu')):
        self.voxel_size = voxel_size
        self.pts_per_shapes = pts_per_shapes
        self.min_surface_pts = min_surface_pts
        self.network = network
        self.device = device
        self.num_aligned = 0
        if network is not None:
            self.network = PointNetTransformer().to(device)
            load_model(network, self.network, device)
            self.network.eval()

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def normalize_mesh(self, mesh):
        return mesh_to_sdf.scale_to_unit_cube(mesh)

    def get_centroid_and_orientation(self, pts):
        random_ind = torch.randperm(
            pts.shape[0])[:self.min_surface_pts]
        voxel_surface = pts[random_ind, :]
        centroid = torch.mean(voxel_surface, dim=0)
        volume_surface = (voxel_surface-centroid) / (1.5*self.voxel_size)
        volume_surface = volume_surface[None, ...].float()
        orientation = self.network(
            volume_surface, transpose_input=True)[0, ...]
        return centroid, orientation

    def gen_samples(self, shape, voxel_id_start, show=False):
        bounding_radius = np.max(np.linalg.norm(shape.vertices, axis=1)) * 2
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(shape)
        points, sdf, surface_points = surface_point_cloud.sample_sdf_near_surface(
            self.pts_per_shapes, sign_method='depth', radius=bounding_radius)

        if show:
            self.display_sdf(points, sdf)

        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        centroids = []
        rotations = []
        with torch.no_grad():
            voxels = torch.from_numpy(voxels).float().to(self.device)
            sdf = torch.from_numpy(sdf).float().to(self.device)
            points = torch.from_numpy(points).float().to(self.device)
            surface_points = torch.from_numpy(
                surface_points).float().to(self.device)
            for vid in range(voxels.shape[0]):
                print("{}/{}".format(vid, voxels.shape[0]))
                voxel_pts = points - voxels[vid, :]
                dist = torch.norm(voxel_pts, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_pts = voxel_pts[selector, :]
                voxel_sdf = sdf[selector]

                voxel_surface = surface_points - voxels[vid, :]
                dist = torch.norm(voxel_surface, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_surface = voxel_surface[selector, :]

                centroid = torch.zeros((3,))
                orientation = torch.eye(3)
                if self.network is not None:
                    if voxel_surface.shape[0] > self.min_surface_pts:
                        centroid, orientation = self.get_centroid_and_orientation(
                            voxel_surface)
                        self.num_aligned += 1
                    else:
                        centroid = torch.mean(voxel_surface, dim=0)

                if show:
                    self.display_sdf(voxel_pts.detach().cpu(
                    ).numpy(), voxel_sdf.detach().cpu().numpy())

                vsample = torch.zeros((voxel_pts.shape[0], 6)).to(self.device)
                vsample[:, 0] = float(vid+voxel_id_start)
                vsample[:, 1:4] = voxel_pts
                vsample[:, 4] = voxel_sdf
                vsample[:, 5] = 1

                samples.append(vsample.detach().cpu().numpy())
                centroids.append(centroid.detach().cpu().numpy())
                rotations.append(orientation.detach().cpu().numpy())
        samples = np.concatenate(samples, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)
        voxels = voxels.detach().cpu().numpy()
        return samples, voxels, centroids, rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--min_surface_pts', type=int, default=2048)
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    model_files = [f for f in os.listdir(
        args.input) if os.path.isfile(os.path.join(args.input, f))]
    os.makedirs(args.output, exist_ok=True)
    num_models = len(model_files)

    num_per_row = 1
    for i in range(1, num_models):
        if i * i >= num_models:
            num_per_row = i
            break

    sampler = MeshSampler(
        args.voxel_size, args.num_samples,
        args.min_surface_pts, args.network,
        device=device)

    all_samples = []
    all_voxels = []
    all_centroids = []
    all_rotations = []
    voxel_id_start = 0

    for i, model in enumerate(model_files):
        mesh = trimesh.load(os.path.join(args.input, model))
        samples, voxels, centroids, rotations = sampler.gen_samples(
            mesh, voxel_id_start, args.show)
        voxel_id_start += voxels.shape[0]
        all_samples.append(samples)
        all_centroids.append(centroids)
        all_rotations.append(rotations)

        y = i // num_per_row
        x = i - y * num_per_row
        all_voxels.append(voxels+np.array([x, y, 0]))

    print("{} voxels sampled with {} aligned".format(
        voxel_id_start, sampler.num_aligned))
        
    out = dict()
    sample_name = 'samples.npy'
    out['samples'] = sample_name
    out['voxels'] = np.concatenate(voxels, axis=0).astype(np.float32)
    out['centroids'] = np.concatenate(centroids, axis=0).astype(np.float32)
    out['rotations'] = np.concatenate(rotations, axis=0).astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
