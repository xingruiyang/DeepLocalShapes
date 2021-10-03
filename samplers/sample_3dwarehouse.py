import sys  # noqa

sys.path.insert(0, '/workspace')  # noqa

import argparse
import os
import pickle
import sys

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from orientation.transformer import PointNetTransformer
from utils.io_utils import load_model

import mesh_to_sdf


def to_np_array(o3d_pcd):
    return np.asarray(o3d_pcd.points)


def to_o3d(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def view_voxels(voxel_points, voxel_sdf, centroid, orientation, voxel_size):

    points = voxel_points - centroid
    points = torch.matmul(points, orientation.transpose(0, 1))
    points = points.detach().cpu().numpy()

    sdf = voxel_sdf.detach().cpu().numpy()
    color = np.zeros_like(points)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    points = to_o3d(points, color)

    bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
        -np.ones((3, )) * .5 * voxel_size,
        np.ones((3, )) * .5 * voxel_size
    )
    bbox_outter = o3d.geometry.AxisAlignedBoundingBox(
        -np.ones((3, )) * 1.5 * voxel_size,
        np.ones((3, )) * 1.5 * voxel_size
    )
    o3d.visualization.draw_geometries([points, bbox_inner, bbox_outter])


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
        bounding_radius = np.max(np.linalg.norm(shape.vertices, axis=1)) * 1.4
        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(shape)
        points, sdf, surface_points = surface_point_cloud.sample_sdf_near_surface(
            self.pts_per_shapes, sign_method='depth', use_scans=True, radius=bounding_radius)
        # points, sdf, surface_points = mesh_to_sdf.sample_sdf_near_surface(
        #     mesh, number_of_points=1500000, surface_point_method='sample')
        if show:
            self.display_sdf(points, sdf)

        voxels = surface_points // self.voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= self.voxel_size

        samples = []
        centroids = []
        rotations = []
        print("{} voxels to be sampled".format(voxels.shape[0]))
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

                # view_voxels(voxel_pts, voxel_sdf, centroid,
                #             orientation, self.voxel_size)

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

    # categories = {
    #     'lamp', 'airplane', 'chair', 'sofa', 'table'
    # }
    categories = {
        'sofa'
    }
    num_per_cat = 10
    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    model_files = [f for f in os.listdir(
        args.input) if os.path.isfile(os.path.join(args.input, f))]
    os.makedirs(args.output, exist_ok=True)
    num_models = len(model_files)
    num_models = len(categories) * num_per_cat
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

    scene = trimesh.Scene()
    for i, cat in enumerate(categories):
        for j, model in enumerate(range(num_per_cat)):
            model_file = os.path.join(args.input, cat, '{}.ply'.format(j))
            mesh = trimesh.load(model_file)
            # verts = mesh.vertices
            # verts *= 0.5 / np.max(np.linalg.norm(verts, axis=-1))
            # mesh = trimesh.Trimesh(verts, mesh.faces)
            transform = np.eye(4)
            transform[:3, :3] = R.random().as_matrix()
            mesh.apply_transform(transform)
            # mesh.show()
            samples, voxels, centroids, rotations = sampler.gen_samples(
                mesh, voxel_id_start, args.show)
            voxel_id_start += voxels.shape[0]
            all_samples.append(samples)
            all_centroids.append(centroids)
            all_rotations.append(rotations)

            ind = i * num_per_cat + j
            y = ind // num_per_row
            x = ind - y * num_per_row
            all_voxels.append(voxels+np.array([x, y, 0]))

            transform = np.eye(4)
            transform[0, 3] = x
            transform[1, 3] = y
            scene.add_geometry(mesh, transform=transform)

    scene = scene.dump(True)
    # scene.show()
    print("{} voxels sampled with {} aligned".format(
        voxel_id_start, sampler.num_aligned))

    out = dict()
    sample_name = 'samples.npy'
    out['samples'] = sample_name
    out['voxels'] = np.concatenate(all_voxels, axis=0).astype(np.float32)
    out['centroids'] = np.concatenate(all_centroids, axis=0).astype(np.float32)
    out['rotations'] = np.concatenate(all_rotations, axis=0).astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    all_samples = np.concatenate(all_samples, axis=0).astype(np.float32)
    np.save(os.path.join(args.output, sample_name), all_samples)
    scene.export(os.path.join(args.output, "gt.ply"))
