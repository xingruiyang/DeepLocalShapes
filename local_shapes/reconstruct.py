# import sys  # noqa
# sys.path.insert(0, '/workspace')  # noqa

import copy
import open3d as o3d
import argparse
import os

import cv2
import numpy as np
import torch
import trimesh as m
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from data import SampleDataset
# from samplers.mesh_to_sdf.mesh_to_sdf import pyrender_wrapper, scan
from network import ImplicitNet
from utils import load_latents, load_model
import pyrender  # noqa


def get_grid_points(resolution, uniform=True, range=[-1, 1], device=torch.device('cpu')):
    if uniform:
        x = y = z = torch.linspace(range[0], range[1], resolution)
        xx, yy, zz = torch.meshgrid(x, y, z)
        grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = grid_points.view(-1, 3).to(device)
        xyz = [x, y, z]
    else:
        raise NotImplementedError()

    return grid_points, xyz


def get_sdf(network, latent_vec, grid_points, batch_size=32**3):
    z = []
    for i, pnts in enumerate(torch.split(grid_points, batch_size, dim=0)):
        latent = latent_vec.expand(pnts.shape[0], -1)
        grid_sample = torch.cat([latent, pnts], dim=-1)
        z.append(network(grid_sample))
    z = torch.cat(z, dim=0)
    return z


def trace_surface_points(z, xyz, mask):
    if not isinstance(z, np.ndarray):
        z = z.detach().cpu().numpy()
    spacing = (xyz[0][2] - xyz[0][1],
               xyz[0][2] - xyz[0][1],
               xyz[0][2] - xyz[0][1])
    try:
        verts, faces, normals, value = marching_cubes(
            volume=z, level=0, spacing=spacing, mask=mask)
        return verts, faces, normals, value
    except:
        return None


class ShapeReconstructor(object):
    def __init__(self,
                 network,
                 latent_vecs,
                 voxels,
                 voxel_size,
                 resolution=8,
                 centroids=None,
                 rotations=None,
                 surface_pts=None,
                 max_surface_dist=0.1,
                 device=torch.device('cpu')):
        super(ShapeReconstructor, self).__init__()
        self.latent_vecs = latent_vecs
        self.voxels = voxels
        self.network = network
        self.voxel_size = voxel_size
        self.device = device
        self.resolution = resolution+1
        self.max_surface_dist = max_surface_dist
        self.surface_pts = surface_pts
        if surface_pts is not None:
            self.surface_pts = NearestNeighbors(
                n_neighbors=1, metric='l2').fit(surface_pts)
        if isinstance(centroids, np.ndarray):
            self.centroids = torch.from_numpy(
                centroids).to(device).float()
        else:
            self.centroids = centroids

        if isinstance(rotations, np.ndarray):
            self.rotations = torch.from_numpy(
                rotations).to(device).float()
        else:
            self.rotations = rotations

    def interp_border(self, z_array, voxels):
        min_voxel = np.round(np.amin(voxels, axis=0)).astype(int)
        max_voxel = np.round(np.amax(voxels, axis=0)).astype(int)
        grid_size = max_voxel - min_voxel + 1
        grid_sdf = np.zeros(grid_size * (self.resolution-1) + 1)
        grid_sdf[:] = 10000

        for i in range(voxels.shape[0]):
            # if i > 100:
            #     break
            z = z_array[i].reshape(
                self.resolution, self.resolution, self.resolution)
            voxel = np.round(voxels[i, :]).astype(int) - min_voxel

            z0 = grid_sdf[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                          voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                          voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]]
            z = np.minimum(z0, z)
            grid_sdf[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                     voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                     voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]] = z

        z_array_interp = []
        for i in range(voxels.shape[0]):
            # if i > 100:
            #     break
            voxel = np.round(voxels[i, :]).astype(int) - min_voxel
            z = grid_sdf[voxel[0] * (self.resolution-1):(voxel[0] + 1) * self.resolution - voxel[0],
                         voxel[1] * (self.resolution-1):(voxel[1] + 1) * self.resolution - voxel[1],
                         voxel[2] * (self.resolution-1):(voxel[2] + 1) * self.resolution - voxel[2]]
            z_array_interp.append(z)
        return z_array_interp

    def reconstruct_interp(self, return_raw=False):
        self.network.eval()
        z_array = []
        z_masks = []
        mesh_verts = []
        mesh_faces = []
        num_exist_verts = 0
        for latent_ind in range(self.voxels.shape[0]):
            grid_pts, xyz = get_grid_points(
                self.resolution, range=[-.5, .5], device=self.device)

            z_mask = None
            if self.surface_pts is not None:
                voxel = self.voxels[latent_ind, :]
                z_mask = self.surface_pts.kneighbors(
                    (grid_pts.detach().cpu().numpy()) * self.voxel_size+voxel)[0]
                z_mask = z_mask.reshape(
                    self.resolution, self.resolution, self.resolution) < self.max_surface_dist

            if self.centroids is not None:
                grid_pts -= self.centroids[latent_ind, :] / self.voxel_size
            if self.rotations is not None:
                grid_pts = torch.matmul(
                    grid_pts, self.rotations[latent_ind, ...].transpose(0, 1))

            latent_vec = self.latent_vecs[latent_ind, :]
            z = get_sdf(self.network, latent_vec, grid_pts)

            z_array.append(z.detach().cpu().numpy())
            z_masks.append(z_mask)
        voxels = self.voxels / self.voxel_size - .5
        z_array = self.interp_border(z_array, voxels)

        for latent_ind in range(self.voxels.shape[0]):
            z = z_array[latent_ind]
            mask = z_masks[latent_ind]
            has_surface = np.min(z) < 0 and np.max(z) > 0
            if has_surface:
                surface = trace_surface_points(z, xyz, mask)
                if surface is None:
                    continue
                verts, faces, _, _ = surface
                verts -= .5
                verts *= self.voxel_size
                verts += self.voxels[latent_ind, :]
                faces += num_exist_verts
                mesh_verts.append(verts)
                mesh_faces.append(faces)
                num_exist_verts += verts.shape[0]

        if len(mesh_verts) == 0:
            return None
        mesh_verts = np.concatenate(mesh_verts, axis=0)
        mesh_faces = np.concatenate(mesh_faces, axis=0)

        if not return_raw:
            return m.Trimesh(mesh_verts, mesh_faces)
        else:
            return m.Trimesh(mesh_verts, mesh_faces), mesh_verts, mesh_faces

        # print(recon_shape.vertex_normals*0.5+0.5)
        # colors = recon_shape.vertex_normals #*0.5+0.5
        # colors[:, :2] *= -1
        # colors = colors * 0.5 + 0.5
        # recon_shape.visual.vertex_colors=colors
        # print(recon_shape.face_normals)
        # recon_shape = m.Trimesh(
        #     mesh_verts, mesh_faces,
        #     face_colors=recon_shape.face_normals*0.5+0.5)
        # return recon_shape

    def reconstruct(self):
        self.network.eval()
        mesh_verts = []
        mesh_faces = []
        num_exist_verts = 0
        for latent_ind in range(self.voxels.shape[0]):
            grid_pts, xyz = get_grid_points(
                self.resolution, range=[-.5, .5], device=self.device)
            grid_pts_origin = copy.deepcopy(grid_pts)
            voxel = self.voxels[latent_ind, :]

            centroid = self.centroids[latent_ind, :]
            rotation = self.rotations[latent_ind, ...]

            z_mask = None
            if self.surface_pts is not None:
                z_mask = self.surface_pts.kneighbors(
                    (grid_pts.detach().cpu().numpy()) * self.voxel_size+voxel)[0]
                z_mask = z_mask.reshape(
                    self.resolution, self.resolution, self.resolution) < self.max_surface_dist

            if self.centroids is not None:
                grid_pts -= (centroid / self.voxel_size)
            if self.rotations is not None:
                grid_pts = torch.matmul(grid_pts, rotation.transpose(0, 1))

            latent_vec = self.latent_vecs[latent_ind, :]
            z = get_sdf(self.network, latent_vec, grid_pts)
            z = z.detach().cpu().numpy()
            has_surface = np.min(z) < 0 and np.max(z) > 0
            if has_surface:
                z = z.reshape(self.resolution,
                              self.resolution,
                              self.resolution)
                surface = trace_surface_points(z, xyz, z_mask)
                verts, faces, _, _ = surface
                # print(np.min(verts), np.max(verts))

                verts -= .5
                # print(np.min(verts), np.max(verts))
                # print(torch.min(grid_pts_origin), torch.max(grid_pts_origin))

                # m.Scene([m.Trimesh(verts, faces),
                #         m.PointCloud((grid_pts_origin.cpu()))]).show()

                verts *= self.voxel_size
                verts += voxel

                faces += num_exist_verts
                mesh_verts.append(verts)
                mesh_faces.append(faces)
                num_exist_verts += verts.shape[0]

        if (len(mesh_verts) == 0):
            print("reconstruction failed.")
            return None
        else:
            mesh_verts = np.concatenate(mesh_verts, axis=0)
            mesh_faces = np.concatenate(mesh_faces, axis=0)
            recon_shape = m.Trimesh(mesh_verts, mesh_faces)
            return recon_shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('latents', type=str)
    parser.add_argument('network', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--latent-size', type=int, default=125)
    parser.add_argument('--max-surface-dist', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--no-interp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')
    net_params = {'d_in': args.latent_size + 3, 'dims': [128, 128, 128]}
    network = ImplicitNet(**net_params).to(device)
    load_model(args.network, network, device)
    latent_vecs = load_latents(args.latents, device)

    dataset = SampleDataset(
        args.data, args.orient, args.crop, training=False)
    reconstructor = ShapeReconstructor(
        network, latent_vecs, dataset.voxels,
        dataset.voxel_size, args.resolution,
        dataset.centroids, dataset.rotations,
        dataset.surface, args.max_surface_dist,
        device=device)

    recon_shape = reconstructor.reconstruct(
    ) if args.no_interp else reconstructor.reconstruct_interp()

    if args.output is not None:
        recon_shape.export(args.output)
    # else:
    #     mesh = pyrender.Mesh.from_trimesh(recon_shape)
    #     scene = pyrender.Scene()
    #     scene.add(mesh)
    #     pyrender.Viewer(scene, viewer_flags={
    #         'use_direct_lighting': True})
    else:
        mesh = recon_shape.as_open3d
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
    # if args.render:
    #     mesh = pyrender.Mesh.from_trimesh(recon_shape)
    #     transform = np.eye(4)
    #     transform[:3, 3] = -np.mean(recon_shape.vertices, axis=0)
    #     recon_shape.apply_transform(transform)
    #     camera = pyrender.PerspectiveCamera(
    #         yfov=1, znear=0.1, zfar=10)
    #     camera_pose = scan.get_camera_transform_looking_at_origin(0, 0, 0.18)
    #     normal, depth = pyrender_wrapper.render_normal_and_depth_buffers(
    #         recon_shape, camera, camera_pose, (640, 480), True)
    #     depth /= np.max(depth)
    #     depth *= 65535
    #     cv2.imwrite("normal.png", normal.astype(np.uint8))
    #     cv2.imwrite("depth.png", depth.astype(np.uint16))
    #     cv2.waitKey(0)
