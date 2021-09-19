import argparse
import glob
import os
import pickle

import cv2
import natsort
import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree

from mesh_to_sdf import mesh_to_sdf


def to_o3d(arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


def to_numpy(arr):
    return np.asarray(arr)


def est_normal(arr, radius, max_nn=30):
    o3d_arr = to_o3d(arr)
    o3d_arr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    return to_numpy(o3d_arr.normals)


class MeshSampler():
    def __init__(self, mesh, num_samples, voxel_size):
        self.mesh = mesh
        self.voxel_size = voxel_size
        self.num_samples = num_samples

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def sample_sdf(self):
        surface_points, face_index = self.mesh.sample(self.num_samples, return_index=True)
        surface_normals = self.mesh.face_normals[face_index]
        dist = 0.01
        points = np.concatenate([
            surface_points,
            surface_points + surface_normals * dist,
            surface_points - surface_normals * dist], axis=0)
        sdf = np.concatenate([
            np.zeros(surface_points.shape[0]),
            np.zeros(surface_points.shape[0]) + dist,
            np.zeros(surface_points.shape[0]) - dist], axis=0)

        self.display_sdf(points, sdf)
        # surface_points = self.mesh.sample(2**16)
        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        with torch.no_grad():
            samples = []
            voxels = torch.from_numpy(voxels).float().cuda()
            sdf = torch.from_numpy(sdf).float().cuda()
            points = torch.from_numpy(points).float().cuda()
            for vid in range(voxels.shape[0]):
                print(vid)
                voxel_pts = points - voxels[vid, :]
                dist = torch.norm(voxel_pts, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_pts = voxel_pts[selector, :]
                voxel_sdf = sdf[selector]
                # self.display_sdf(voxel_pts, voxel_sdf)
                vsample = torch.zeros((voxel_pts.shape[0], 6)).cuda()
                vsample[:, 0] = float(vid)
                vsample[:, 1: 4] = voxel_pts
                vsample[:, 4] = voxel_sdf
                vsample[:, 5] = 1
                vsample = vsample
                samples.append(vsample.cpu().numpy())
            samples = np.concatenate(samples, axis=0)
            return samples, voxels.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=250000)
    args = parser.parse_args()

    print("Using mesh sampler")
    mesh = trimesh.load(args.input)
    sampler = MeshSampler(mesh, args.num_samples, args.voxel_size)

    samples, voxels = sampler.sample_sdf()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    out['voxels'] = voxels
    out['samples'] = samples
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "sample.pkl"), "wb") as f:
        pickle.dump(out, f)
