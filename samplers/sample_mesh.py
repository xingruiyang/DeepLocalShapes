import sys

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
                 pts_per_voxel=2048,
                 network=None,
                 normalize=False,
                 device=torch.device('cpu')):
        self.mesh = mesh
        self.voxel_size = voxel_size
        self.pts_per_voxel = pts_per_voxel
        self.network = network
        if network is not None:
            self.network = PointNetTransformer().to(device)
            load_model(network, self.network, device)
            self.network.eval()
        self.device = device
        self.num_aligned = 0
        self.normalize = normalize

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def get_orientation(self, pts):
        volumes = pts / (1.5 * self.voxel_size)
        volumes = torch.from_numpy(volumes)[None, ...].float()
        rotation = self.network(volumes.to(self.device), transpose_input=True)
        return rotation.suqeeze().detach().cpu().numpy()

    # def get_sdf(self, kd_tree, pcd, normal, points):
    #     sample_count = 11
    #     sdf, indices = kd_tree.query(points, k=sample_count)
    #     closest_points = pcd[indices]
    #     dir_from_surface = points[:, None, :] - closest_points
    #     inside = np.einsum('ijk,ijk->ij', dir_from_surface,
    #                        normal[indices]) < 0
    #     inside = np.sum(inside, axis=1) > (sample_count * 0.6)
    #     sdf = sdf[:, 0]
    #     sdf[inside] *= -1
    #     return sdf

    def sample_sdf(self, show=False):
        # bounding_radius = np.max(np.linalg.norm(
        #     self.mesh.vertices, axis=1)) * 1.4
        surface_points = mesh_to_sdf.get_surface_point_cloud(
            self.mesh, scan_count=100)
        # points, sdf, surface_points = surface_point_cloud.sample_sdf_near_surface(
        #     args.num_samples, sign_method='depth', use_scans=True, radius=bounding_radius)

        # if show:
        #     self.display_sdf(points, sdf)

        voxels = surface_points.points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= self.voxel_size
        print("{} voxels to be sampled".format(voxels.shape[0]))

        # kd_tree = KDTree(surface_points.points)

        samples = []
        centroids = []
        rotations = []

        surface = []
        rand_surface = []
        rand_sdf = []

        for vid in range(voxels.shape[0]):
            print("{}/{}".format(vid, voxels.shape[0]))
            voxel = voxels[vid, :]
            pcd = surface_points.points - voxel
            dist = np.linalg.norm(pcd, ord=np.inf, axis=-1)
            selector = dist < (1.5 * self.voxel_size)
            pcd = pcd[selector, :]

            if pcd.shape[0] > args.pts_per_voxel:
                pcd = pcd[np.random.permutation(
                    pcd.shape[0])[:args.pts_per_voxel], :]

            centroid = np.mean(pcd, axis=0)
            rotation = np.eye(3)
            if self.network is not None and pcd.shape[0] == args.pts_per_voxel:
                rotation = self.get_orientation(pcd-centroid)
                self.num_aligned += 1

            query_points = []
            query_points.append(pcd+np.random.randn(*pcd.shape)*0.0025)
            query_points.append(pcd+np.random.randn(*pcd.shape)*0.00025)
            query_points.append(
                (np.random.rand(1024, 3)*3-1.5)*self.voxel_size)
            query_points = np.concatenate(query_points, axis=0)

            sdf = surface_points.get_sdf(query_points + voxel)
            
            surface.append(pcd+voxel)
            rand_surface.append(query_points[:pcd.shape[0]*2, :]+voxel)
            rand_sdf.append(sdf[:pcd.shape[0]*2])

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

        surface = np.concatenate(surface, axis=0)
        trimesh.PointCloud(surface).show()
        rand_surface = np.concatenate(rand_surface, axis=0)
        rand_sdf = np.concatenate(rand_sdf, axis=0)
        self.display_sdf(rand_surface, rand_sdf)

        return samples, voxels, centroids, rotations

        # with torch.no_grad():
        #     voxels = torch.from_numpy(voxels).float().to(self.device)
        #     # sdf = torch.from_numpy(sdf).float().to(self.device)
        #     points = torch.from_numpy(points).float().to(self.device)
        #     surface_points = torch.from_numpy(
        #         surface_points).float().to(self.device)
        #     for vid in range(voxels.shape[0]):
        #         print("{}/{}".format(vid, voxels.shape[0]))
        #         voxel_pts = points - voxels[vid, :]
        #         dist = torch.norm(voxel_pts, p=np.inf, dim=-1)
        #         selector = dist < (1.5 * self.voxel_size)
        #         voxel_pts = voxel_pts[selector, :]
        #         voxel_sdf = sdf[selector]

        #         voxel_surface = surface_points - voxels[vid, :]
        #         dist = torch.norm(voxel_surface, p=np.inf, dim=-1)
        #         selector = dist < (1.5 * self.voxel_size)
        #         voxel_surface = voxel_surface[selector, :]

        #         centroid = torch.zeros((3,))
        #         orientation = torch.eye(3)
        #         if self.network is not None:
        #             if voxel_surface.shape[0] > self.pts_per_voxel:
        #                 centroid, orientation = self.get_centroid_and_orientation(
        #                     voxel_surface)
        #                 self.num_aligned += 1
        #             else:
        #                 centroid = torch.mean(voxel_surface, dim=0)

        #         # view_voxels(voxel_pts, voxel_sdf, centroid,
        #         #             orientation, self.voxel_size)

        #         if show:
        #             self.display_sdf(voxel_pts.detach().cpu(
        #             ).numpy(), voxel_sdf.detach().cpu().numpy())

        #         vsample = torch.zeros((voxel_pts.shape[0], 6)).to(self.device)
        #         vsample[:, 0] = float(vid)
        #         vsample[:, 1:4] = voxel_pts
        #         vsample[:, 4] = voxel_sdf
        #         vsample[:, 5] = 1

        #         samples.append(vsample.detach().cpu().numpy())
        #         centroids.append(centroid.detach().cpu().numpy())
        #         rotations.append(orientation.detach().cpu().numpy())
        # samples = np.concatenate(samples, axis=0)
        # centroids = np.stack(centroids, axis=0)
        # rotations = np.stack(rotations, axis=0)
        # voxels = voxels.detach().cpu().numpy()
        # return samples, voxels, centroids, rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--pts_per_voxel', type=int, default=4096)
    # parser.add_argument('--num_samples', type=int, default=1500000)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    mesh = trimesh.load(args.input)
    sampler = MeshSampler(
        mesh, args.voxel_size,
        args.pts_per_voxel,
        args.network,
        args.normalize,
        device=device)

    samples, voxels, centroids, rotations = sampler.sample_sdf(args.show)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    sample_name = 'samples.npy'
    out['samples'] = sample_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name), samples.astype(np.float32))
