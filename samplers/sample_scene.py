import sys  # noqa

sys.path.insert(0, '/workspace')  # noqa
import argparse
import glob
import os
import pickle

import cv2
import natsort
import numpy as np
import torch
import trimesh
from orientation.transformer import PointNetTransformer
from sklearn.neighbors import KDTree
from utils.io_utils import load_model

_EPS = np.finfo(float).eps * 4.0


class DepthSampler():
    def __init__(
            self,
            scene_path: str,
            downsample: int,
            skip_frames: int,
            voxel_size: float,
            depth_limit: float,
            min_surface_pts: int = 2048,
            transformer=None,
            device=torch.device('cpu')):
        self.scene_path = scene_path
        self.downsample = downsample
        self.voxel_size = voxel_size
        self.skip_frames = skip_frames
        self.depth_limit = depth_limit
        self.transformer = transformer
        self.min_surface_pts = min_surface_pts
        self.device = device
        if transformer is not None:
            self.transformer = PointNetTransformer().to(device)
            load_model(transformer, self.transformer, device)
            self.transformer.eval()

    def get_voxels(self, pts):
        voxels = pts // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size
        return voxels

    def read_file_list(self, filename):
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                for line in lines if len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
        return dict(list)

    def display_sdf(self, pts, sdf):
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        if isinstance(sdf, torch.Tensor):
            sdf = sdf.detach().cpu().numpy()
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

    def get_centroid_and_orientation(self, pts):
        random_ind = torch.randperm(pts.shape[0])[:self.min_surface_pts]
        voxel_surface = pts[random_ind, :]
        centre = torch.mean(voxel_surface, dim=0)
        volume_surface = (voxel_surface-centre) / (1.5*self.voxel_size)
        volume_surface = volume_surface[None, ...].float()
        orientation = self.transformer(
            volume_surface, transpose_input=True)
        return centre, orientation[0, ...]

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
        depth = depth / depth_scale
        depth[depth > 10] = 0
        if down_scale != 0:
            shape = [s // 2**down_scale for s in depth.shape]
            depth = cv2.resize(
                depth, shape[::-1], interpolation=cv2.INTER_NEAREST)
        return depth

    def transform44(self, l):
        t = l[1:4]
        q = np.array(l[4:8], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < _EPS:
            return np.array([
                [1.0, 0.0, 0.0, t[0]],
                [0.0, 1.0, 0.0, t[1]],
                [0.0, 0.0, 1.0, t[2]],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0-q[1, 1]-q[2, 2], q[0, 1]-q[2, 3], q[0, 2]+q[1, 3], t[0]),
            (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2], q[1, 2]-q[0, 3], t[1]),
            (q[0, 2]-q[1, 3], q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)

    def read_trajectory(self, filename, matrix=True, offset=0):
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""]
                for line in lines if len(line) > 0 and line[0] != "#"]
        list_ok = []
        for i, l in enumerate(list):
            if l[4:8] == [0, 0, 0, 0]:
                continue
            isnan = False
            for v in l:
                if np.isnan(v):
                    isnan = True
                    break
            if isnan:
                sys.stderr.write(
                    "Warning: line %d of file '%s' has NaNs, skipping line\n" % (i, filename))
                continue
            list_ok.append(l)
        if matrix:
            traj = dict([(l[0]+offset, self.transform44(l[0:]))
                        for l in list_ok])
        else:
            traj = dict([(l[0]+offset, l[1:8]) for l in list_ok])
        return traj

    def sample_sdf(self):
        calib = os.path.join(self.scene_path, 'calib/calib.txt')
        calib = np.loadtxt(calib)
        intr = np.eye(3)
        intr[0, 0] = calib[0]
        intr[1, 1] = calib[1]
        intr[0, 2] = calib[2]
        intr[1, 2] = calib[3]
        intr //= (2**self.downsample)
        intr[2, 2] = 1
        depth_scale = calib[4]
        inv_y_axis = calib[1] < 0

        assoc = self.read_file_list(os.path.join(
            self.scene_path, 'assoc/assoc.txt'))
        trajectory = self.read_trajectory(
            os.path.join(self.scene_path, 'groundtruth.txt'))

        surface_points = []
        surface_normals = []
        free_space_samples = []
        point_weights = []
        rand_weights = []
        for ind, (key, val) in enumerate(assoc.items()):
            if ind % args.skip_frames != 0:  # or ind >= 5:
                continue

            pose = trajectory[key]
            depth_filename = os.path.join(self.scene_path, val[2])
            depth = self.load_depth_map(
                depth_filename, self.downsample, depth_scale)

            pcd, rays = self.get_point_cloud(depth, intr, True)
            normal = self.get_normal_map(pcd, inv_y_axis).reshape(-1, 3)
            depth = depth.reshape(-1, 1)
            pcd = pcd.reshape(-1, 3)
            rays = rays.reshape(-1, 3)
            nonzeros = np.logical_and(depth[:, 0] != 0, normal[:, 2] != 0)
            depth = depth[nonzeros, :]
            rays = rays[nonzeros, :]
            pcd = pcd[nonzeros, :]
            normal = normal[nonzeros, :]
            normal = np.matmul(
                normal, pose[:3, :3].transpose())
            pcd = np.matmul(
                pcd, pose[:3, :3].transpose()) + pose[:3, 3]
            rand_pts = 0.985-(np.random.rand(rays.shape[0], 1)*0.4)
            rand_pts = rays * rand_pts * depth
            rand_pts = np.matmul(
                rand_pts, pose[:3, :3].transpose()) + pose[:3, 3]

            weight = 1.0 / depth
            randpick = np.random.permutation(rand_pts.shape[0])[
                :rand_pts.shape[0]//8]
            rand_pts = rand_pts[randpick, :]
            rand_weight = weight[randpick, :]

            free_space_samples.append(rand_pts)
            point_weights.append(weight)
            rand_weights.append(rand_weight)
            surface_points.append(pcd)
            surface_normals.append(normal)
        point_weights = np.concatenate(point_weights, axis=0)
        surface_points = np.concatenate(surface_points, axis=0)
        surface_normals = np.concatenate(surface_normals, axis=0)
        free_space_weights = np.concatenate(rand_weights, axis=0)
        free_space_samples = np.concatenate(free_space_samples, axis=0)

        print(
            "contstructing KD-Tree with {} points".format(surface_points.shape[0]))
        kd_tree = KDTree(surface_points)
        print(
            "querying KD-Tree with {} points".format(free_space_samples.shape[0]))
        free_space_sdf, _ = kd_tree.query(free_space_samples)
        free_space_sdf = free_space_sdf[:, 0]

        dist1 = 0.01
        dist2 = 0.005
        points = np.concatenate([
            surface_points,
            surface_points + surface_normals * dist1,
            surface_points - surface_normals * dist1,
            surface_points + surface_normals * dist2,
            surface_points - surface_normals * dist2,
            free_space_samples
        ], axis=0)
        sdf = np.concatenate([
            np.zeros(surface_points.shape[0]),
            np.zeros(surface_points.shape[0]) + dist1,
            np.zeros(surface_points.shape[0]) - dist1,
            np.zeros(surface_points.shape[0]) + dist2,
            np.zeros(surface_points.shape[0]) - dist2,
            free_space_sdf
        ], axis=0)
        weights = np.concatenate([
            point_weights,
            point_weights,
            point_weights,
            point_weights,
            point_weights,
            free_space_weights
        ], axis=0).squeeze()
        print(weights.shape)
        print("We get {} points".format(points.shape[0]))
        self.display_sdf(points, sdf)

        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        centroids = []
        rotations = []
        measurements = []
        num_oriented_voxels = 0
        with torch.no_grad():
            voxels = torch.from_numpy(voxels).float().to(self.device)
            sdf = torch.from_numpy(sdf).float().to(self.device)
            points = torch.from_numpy(points).float().to(self.device)
            weights = torch.from_numpy(weights).float().to(self.device)
            surface_points = torch.from_numpy(
                surface_points).float().to(self.device)
            for vid in range(voxels.shape[0]):
                print("{}/{}".format(vid, voxels.shape[0]))
                voxel_pts = points - voxels[vid, :]
                dist = torch.norm(voxel_pts, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_pts = voxel_pts[selector, :]
                voxel_sdf = sdf[selector]
                voxel_weights = weights[selector]

                voxel_surface = surface_points - voxels[vid, :]
                dist = torch.norm(voxel_surface, p=np.inf, dim=-1)
                selector = dist < (1.5 * self.voxel_size)
                voxel_surface = voxel_surface[selector, :]

                centroid = torch.zeros((3,))
                orientation = torch.eye(3)
                if self.transformer is not None:
                    if voxel_surface.shape[0] > self.min_surface_pts:
                        centroid, orientation = self.get_centroid_and_orientation(
                            voxel_surface)
                        num_oriented_voxels += 1
                    else:
                        centroid = torch.mean(voxel_surface, dim=0)
                centroids.append(centroid.detach().cpu().numpy())
                rotations.append(orientation.detach().cpu().numpy())
                # self.display_sdf(voxel_pts, voxel_sdf)

                vsample = torch.zeros((voxel_pts.shape[0], 6)).to(self.device)
                vsample[:, 0] = float(vid)
                vsample[:, 1:4] = voxel_pts
                vsample[:, 4] = voxel_sdf
                vsample[:, 5] = voxel_weights
                samples.append(vsample.detach().cpu().numpy())
                # measurements.append(voxel_surface.deatch().cpu().numpy())
                #np.save(os.path.join(args.output, "{}.npy".format(vid)), vsample)

        print("total of {} voxels sampled with {} oriented".format(
            voxels.shape[0], num_oriented_voxels))
        samples = np.concatenate(samples, axis=0)
        surface_points = surface_points.detach().cpu().numpy()
        return samples, voxels, centroids, rotations, surface_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--depth_limit', type=float, default=10)
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--skip_frames', type=int, default=10)
    parser.add_argument('--min_surface_pts', type=int, default=2048)
    parser.add_argument('--transformer', type=str, default=None)
    parser.add_argument("--cpu", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')

    sampler = DepthSampler(
        args.input, args.downsample,
        args.skip_frames, args.voxel_size,
        args.depth_limit, args.min_surface_pts,
        args.transformer, device)

    samples, voxels, centroids, rotations, surface_points = sampler.sample_sdf()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    out = dict()
    sample_name = 'samples.npy'
    measurement_name = 'surface.npy'
    out['samples'] = sample_name
    out['surface'] = measurement_name
    out['voxels'] = voxels.detach().cpu().numpy().astype(np.float32)
    out['centroids'] = np.stack(centroids, axis=0).astype(np.float32)
    out['rotations'] = np.stack(rotations, axis=0).astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
    np.save(os.path.join(args.output, measurement_name),
            surface_points.astype(np.float32))
