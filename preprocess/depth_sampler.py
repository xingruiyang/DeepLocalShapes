import argparse
import glob
import os

import cv2
import natsort
import numpy as np
import torch
import trimesh


class DepthSampler(object):
    def __init__(self,
                 scene_path,
                 downsample,
                 skip_frames,
                 depth_limit=10,
                 rand_smaple=False):
        super(DepthSampler, self).__init__()
        self.scene_path = scene_path
        self.downsample = downsample
        self.skip_frames = skip_frames
        self.depth_limit = depth_limit
        self.rand_sample = rand_smaple

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

    def get_normal_map(self, pcd, inv_y_axis=False):
        dx = np.zeros_like(pcd)
        dy = np.zeros_like(pcd)
        dx[1:-1, ...] = pcd[2:, :, :] - pcd[:-2, :, :]
        dy[:, 1:-1, :] = pcd[:, 2:, :] - pcd[:, :-2, :]
        ldx = np.linalg.norm(dx, axis=-1)
        ldy = np.linalg.norm(dx, axis=-1)
        if inv_y_axis:
            normal = np.cross(dy, dx, axis=-1)
        else:
            normal = np.cross(dx, dy, axis=-1)
        normal = self.normalized(normal)
        normal[-1, :, 2] = 0
        normal[0, :, 2] = 0
        normal[:, -1, 2] = 0
        normal[:, 0, 2] = 0
        normal[np.logical_or(ldx > 0.05, ldy > 0.05)] = 0
        return normal

    def load_depth_map(self, depth_path, down_scale=0, depth_scale=1000.0):
        depth = cv2.imread(depth_path, -1)
        depth[depth > 10000] = 0
        shape = [s // 2**down_scale for s in depth.shape]
        depth = depth.astype(np.float32) / depth_scale
        depth = cv2.resize(depth, shape[::-1], interpolation=cv2.INTER_NEAREST)
        return depth

    def sample_sdf(self):
        intr_path = os.path.join(self.scene_path, 'camera-intrinsics.txt')
        intr = np.loadtxt(intr_path)
        intr //= (2**self.downsample)
        intr[2, 2] = 1
        depth_files = glob.glob(os.path.join(
            self.scene_path,  "seq-01/*.depth.png"))
        depth_files = natsort.natsorted(depth_files)

        surface_points = []
        surface_normals = []
        point_weights = []
        random_pnts = []
        for index in range(0, len(depth_files), self.skip_frames):
            filepath = depth_files[index]
            pose_path = os.path.join(
                self.scene_path, 'seq-01/frame-{:06d}.pose.txt'.format(index))
            pose = np.loadtxt(pose_path).astype(float)
            depth = self.load_depth_map(filepath, self.downsample, 1000)

            pcd = self.get_point_cloud(depth, intr)
            normal = self.get_normal_map(pcd).reshape(-1, 3)
            depth = depth.reshape(-1, 1)
            pcd = pcd.reshape(-1, 3)
            nonzeros = np.logical_and(depth[:, 0] != 0, normal[:, 2] != 0)
            depth = depth[nonzeros, :]
            pcd = pcd[nonzeros, :]
            normal = normal[nonzeros, :]
            normal = np.matmul(
                normal, pose[:3, :3].transpose())
            pcd = np.matmul(
                pcd, pose[:3, :3].transpose()) + pose[:3, 3]

            point_weights.append(1.0/depth)
            surface_points.append(pcd)
            surface_normals.append(normal)

        point_weights = np.concatenate(point_weights, axis=0)
        surface_points = np.concatenate(surface_points, axis=0)
        surface_normals = np.concatenate(surface_normals, axis=0)
        point_cloud = np.concatenate(
            [point_weights, surface_points, surface_normals], axis=-1)
        return point_cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_path', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--depth-limit', type=float, default=10)
    args = parser.parse_args()

    sampler = DepthSampler(
        args.scene_path, args.downsample,
        args.skip_frames, args.depth_limit)
    point_cloud = sampler.sample_sdf()

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'pcd.npy'), point_cloud)