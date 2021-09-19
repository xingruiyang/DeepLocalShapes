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
    def __init__(self, scene_path, downsample, skip_frames, voxel_size, depth_limit):
        self.scene_path = scene_path
        self.downsample = downsample
        self.voxel_size = voxel_size
        self.skip_frames = skip_frames
        self.depth_limit = depth_limit

    def get_voxels(self, pts):
        voxels = pts // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size
        return voxels

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def get_point_cloud(self, depth, intr, return_template=False):
        pcd_template = np.ones((depth.shape[0], depth.shape[1], 3))
        pcd_template[:, :, 0], pcd_template[:, :, 1] = np.meshgrid(
            np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        pcd_template = np.matmul(
            pcd_template, np.transpose(np.linalg.inv(intr)))
        pcd = depth[..., None]*pcd_template
        if return_template:
            return pcd, pcd_template
        else:
            return pcd

    def normalized(self, a, axis=-1, order=2):
        l2 = np.linalg.norm(a, order, axis)
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_normal_map(self, pcd):
        dx = np.zeros_like(pcd)
        dy = np.zeros_like(pcd)
        dx[:-1, ...] = pcd[1:, :, :] - pcd[:-1, :, :]
        dy[:, :-1, :] = pcd[:, 1:, :] - pcd[:, :-1, :]
        normal = np.cross(dx, dy, axis=-1)
        normal = self.normalized(normal)
        return normal

    def load_depth_map(self, depth_path, down_scale=0, depth_scale=1000.0):
        depth = cv2.imread(depth_path, -1)
        depth[depth > depth_scale*10] = 0
        depth = depth.astype(np.float32) / depth_scale
        if down_scale > 0:
            shape = [s // 2**down_scale for s in depth.shape]
            depth = cv2.resize(
                depth, shape[::-1], interpolation=cv2.INTER_NEAREST)
        return depth

    def sample_sdf(self):
        intr_path = os.path.join(self.scene_path, 'camera-intrinsics.txt')
        intr = np.loadtxt(intr_path)
        intr //= (2**self.downsample)
        intr[2, 2] = 1
        depth_files = glob.glob(os.path.join(
            self.scene_path,  "seq-01/*.depth.png"))
        depth_files = natsort.natsorted(depth_files)
        
        point_weights = []
        surface_points = []
        surface_normals = []
        free_space_samples = []
        for index in range(0, len(depth_files), args.skip_frames):
            filepath = depth_files[index]
            pose_path = os.path.join(
                self.scene_path, 'seq-01/frame-{:06d}.pose.txt'.format(index))
            pose = np.loadtxt(pose_path).astype(float)
            depth = self.load_depth_map(filepath, self.downsample, 1000)

            pcd, rays = self.get_point_cloud(depth, intr, True)
            normal = self.get_normal_map(pcd).reshape(-1, 3)
            depth = depth.reshape(-1, 1)
            pcd = pcd.reshape(-1, 3)
            rays = rays.reshape(-1, 3)
            nonzeros = np.logical_and(
                pcd[:, 2] != 0, pcd[:, 2] < self.depth_limit)
            depth = depth[nonzeros, :]
            rays = rays[nonzeros, :]
            pcd = pcd[nonzeros, :]
            normal = normal[nonzeros, :]
            normal = np.matmul(
                normal, pose[:3, :3].transpose())
            pcd = np.matmul(
                pcd, pose[:3, :3].transpose()) + pose[:3, 3]
            # normal = est_normal(pcd, radius=self.voxel_size, max_nn=100)
            # print(np.linalg.norm(normal,axis=-1))

            samples = 0.985-(np.random.rand(rays.shape[0], 1)*0.4)
            samples = rays * samples * depth
            samples = np.matmul(
                samples, pose[:3, :3].transpose()) + pose[:3, 3]

            free_space_samples.append(samples)
            point_weights.append(1.0/depth)
            surface_points.append(pcd)
            surface_normals.append(normal)

        point_weights = np.concatenate(point_weights, axis=0)
        surface_points = np.concatenate(surface_points, axis=0)
        surface_normals = np.concatenate(surface_normals, axis=0)
        free_space_samples = np.concatenate(free_space_samples, axis=0)
        # surface_normals = est_normal(surface_points, radius=self.voxel_size, max_nn=50)

        kd_tree = KDTree(surface_points)
        free_space_sdf, _ = kd_tree.query(free_space_samples)
        free_space_sdf = free_space_sdf[:, 0]

        dist = 0.015
        points = np.concatenate([
            # surface_points,
            surface_points + surface_normals * dist,
            surface_points - surface_normals * dist,
            free_space_samples], axis=0)
        sdf = np.concatenate([
            # np.zeros(surface_points.shape[0]),
            np.zeros(surface_points.shape[0]) + dist,
            np.zeros(surface_points.shape[0]) - dist,
            free_space_sdf], axis=0)
        weights = np.concatenate([
            # point_weights, 
            point_weights,
            point_weights, 
            point_weights], axis=0)
        print(points.shape)
        self.display_sdf(points, sdf)

        voxels = self.get_voxels(surface_points)
        with torch.no_grad():
            samples = []
            voxels = torch.from_numpy(voxels).float().cuda()
            sdf = torch.from_numpy(sdf).float().cuda()
            points = torch.from_numpy(points).float().cuda()
            weights = torch.from_numpy(weights).float().cuda()
            for vid in range(voxels.shape[0]):
                print(vid)
                voxel_pts = points - voxels[vid, :]
                dist = torch.norm(voxel_pts, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_pts = voxel_pts[selector, :]
                voxel_sdf = sdf[selector]
                voxel_weight = weights[selector, 0]
                # self.display_sdf(voxel_pts, voxel_sdf)
                vsample = torch.zeros((voxel_pts.shape[0], 6)).cuda()
                vsample[:, 0] = float(vid)
                vsample[:, 1: 4] = voxel_pts
                vsample[:, 4] = voxel_sdf
                vsample[:, 5] = voxel_weight
                vsample = vsample
                samples.append(vsample.cpu().numpy())
            samples = np.concatenate(samples, axis=0)
            return samples, voxels.cpu().numpy()


# 683886
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--depth_limit', type=float, default=10)
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--skip_frames', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1000000)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        print("Using mesh sampler")
        mesh = trimesh.load(args.input)
        sampler = MeshSampler(mesh, args.voxel_size)
    else:
        sampler = DepthSampler(
            args.input, args.downsample,
            args.skip_frames, args.voxel_size,
            args.depth_limit)

    samples, voxels = sampler.sample_sdf()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    out['voxels'] = voxels
    out['samples'] = samples
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "sample.pkl"), "wb") as f:
        pickle.dump(out, f)
