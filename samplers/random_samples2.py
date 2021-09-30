import sys  # noqa

sys.path.insert(0, '/workspace')  # noqa

import argparse
import os
import pickle
import random

import numpy as np
import torch
import trimesh
from orientation.transformer import PointNetTransformer
from scipy.spatial.transform import Rotation as R
from utils.io_utils import load_model

import mesh_to_sdf


class SampleGenerator(object):
    def __init__(self,
                 num_shapes=100,
                 voxel_size=0.1,
                 pts_per_shapes=100000,
                 network=None,
                 min_surface_pts=2048,
                 device=torch.device('cpu')):
        self.network = None
        self.device = device
        self.num_shapes = num_shapes
        self.voxel_size = voxel_size
        self.scene = None
        self.min_surface_pts = min_surface_pts
        self.pts_per_shapes = pts_per_shapes
        self.shape_type = [
            'cuboid',  'cylinder',  'ellipsoid']
        if network is not None:
            self.network = PointNetTransformer().to(device)
            load_model(network, self.network)
            self.network.eval()

        self.total_voxels = 0
        self.num_oriented = 0

    def random_shapes(self, stype='box'):
        if stype == 'cuboid':
            return self.random_cuboid()
        if stype == 'cylinder':
            return self.random_cylinder()
        if stype == 'cone':
            return self.random_cone()
        if stype == 'sphere':
            return self.random_sphere()
        if stype == 'ellipsoid':
            return self.random_ellipsoid()

    def random_cuboid(self):
        return trimesh.creation.box(
            extents=[
                random.uniform(0.1, 1),
                random.uniform(0.1, 1),
                random.uniform(0.1, 1),
            ])

    def random_cylinder(self):
        return trimesh.creation.cylinder(
            radius=random.uniform(0.1, 0.3),
            height=random.uniform(0.1, 1))

    def random_cone(self):
        return trimesh.creation.cone(
            radius=random.uniform(0.1, 0.3),
            height=random.uniform(0.1, 1))

    def random_sphere(self):
        return trimesh.primitives.Sphere(
            radius=random.uniform(0.1, 0.3))

    def random_ellipsoid(self):
        sphere = trimesh.primitives.Sphere(radius=1)
        verts = sphere.vertices
        new_verts = np.zeros_like(verts)
        new_verts[:, 0] = verts[:, 0] * random.uniform(1, 5)
        new_verts[:, 1] = verts[:, 1] * random.uniform(1, 5)
        new_verts[:, 2] = verts[:, 2] * random.uniform(1, 5)
        return trimesh.Trimesh(new_verts, sphere.faces)

    def normalize_shape(self, shape):
        vertices = shape.vertices - shape.bounding_box.centroid
        vertices *= 0.2 / np.max(shape.bounding_box.extents)
        return trimesh.Trimesh(vertices=vertices, faces=shape.faces)

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

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

    def gen_samples(self, shape, voxel_id_start=0, show=False):
        bounding_radius = np.max(np.linalg.norm(shape.vertices, axis=1)) * 1.4
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(shape)
        points, sdf, surface_points = surface_point_cloud.sample_sdf_near_surface(
            self.pts_per_shapes, sign_method='normal', radius=bounding_radius)

        if show:
            self.display_sdf(points, sdf)

        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += 0.5
        voxels *= self.voxel_size

        samples = []
        centroids = []
        rotations = []
        # num_oriented_voxels = 0
        with torch.no_grad():
            voxels = torch.from_numpy(voxels).float().to(self.device)
            sdf = torch.from_numpy(sdf).float().to(self.device)
            points = torch.from_numpy(points).float().to(self.device)
            surface_points = torch.from_numpy(
                surface_points).float().to(self.device)
            for vid in range(voxels.shape[0]):
                # print("{}/{}".format(vid, voxels.shape[0]))
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
                        self.num_oriented += 1
                    else:
                        centroid = torch.mean(voxel_surface, dim=0)

                if show:
                    self.display_sdf(voxel_pts.detach().cpu(
                    ).numpy(), voxel_sdf.detach().cpu().numpy())
                # print(voxel_pts.shape[0])

                vsample = torch.zeros((voxel_pts.shape[0], 6)).to(self.device)
                vsample[:, 0] = float(vid)+voxel_id_start
                vsample[:, 1:4] = voxel_pts
                vsample[:, 4] = voxel_sdf
                vsample[:, 5] = 1

                samples.append(vsample.detach().cpu().numpy())
                centroids.append(centroid.detach().cpu().numpy())
                rotations.append(orientation.detach().cpu().numpy())
            # print("total of {} voxels sampled with {} oriented".format(
            #     voxels.shape[0], num_oriented_voxels))
        samples = np.concatenate(samples, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)
        voxels = voxels.detach().cpu().numpy()
        print(centroids.shape)
        return samples, voxels, centroids, rotations

    def generate_samples(self):
        scene = trimesh.Scene()
        for i in range(self.num_shapes):
            print("processing shape {}/{}".format(i, self.num_shapes))
            stype = random.choice(self.shape_type)
            shape = self.random_shapes(stype)
            shape = self.normalize_shape(shape)
            transform = np.eye(4)
            transform[:3, :3] = R.random().as_matrix()
            shape.apply_transform(transform)

            transform = np.eye(4)
            transform[0, 3] = np.random.rand()*2-1
            transform[1, 3] = np.random.rand()*2-1
            transform[2, 3] = np.random.rand()*2-1
            scene.add_geometry(shape, transform=transform)
        scene.show()
        self.scene = scene

    def save_samples(self, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        scene = self.scene.dump(True)
        scene.export(os.path.join(out_path, "gt.ply"))
        samples, voxels, centroids, rotations = self.gen_samples(
            scene, show=True)
        out = dict()
        sample_name = 'samples.npy'
        out['samples'] = sample_name
        out['voxels'] = voxels.astype(np.float32)
        out['centroids'] = centroids.astype(np.float32)
        out['rotations'] = rotations.astype(np.float32)
        out['voxel_size'] = args.voxel_size
        with open(os.path.join(out_path, "samples.pkl"), "wb") as f:
            pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
        np.save(os.path.join(out_path, sample_name),
                samples.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--num-shapes', type=int, default=10)
    parser.add_argument('--min-surf-pts', type=int, default=2048)
    parser.add_argument('--pts-per-shape', type=int, default=1500000)
    parser.add_argument("--cpu", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    generator = SampleGenerator(
        args.num_shapes, args.voxel_size,
        args.pts_per_shape, args.network,
        args.min_surf_pts, device=device)
    generator.generate_samples()
    generator.save_samples(args.output)
