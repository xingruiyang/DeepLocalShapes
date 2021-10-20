import trimesh
import open3d as o3d
import argparse

import numpy as np
import torch
from skimage.measure import marching_cubes

from deep_sdf.dataset import SampleDataset
from deep_sdf.network import ImplicitNet


def has(arg):
    return arg is not None


class VoxelGrid():
    def __init__(
            self,
            coords: np.ndarray,
            voxel_size: float,
            centroids: np.ndarray = None,
            rotations: np.ndarray = None
    ) -> None:
        self.coords = coords
        self.voxel_size = voxel_size
        self.centroids = centroids
        self.rotations = rotations
        self.num_voxels = coords.shape[0]

    def size(self):
        return self.num_voxels

    def get_local_grid(
            self,
            ind: int = 0,
            v_res: int = 8,
            range: list = [-1, 1],
            canonical: bool = True,
            return_spacing: bool = False):

        x = y = z = np.linspace(range[0], range[1], v_res)
        xx, yy, zz = np.meshgrid(x, y, z)
        vgrid_pts = np.stack([xx, yy, zz], axis=-1)
        vgrid_pts = vgrid_pts.reshape(-1, 3)

        if has(self.centroids) and canonical:
            vgrid_pts -= self.centroids[ind, ...] / (1.5*self.voxel_size)
        if has(self.rotations) and canonical:
            vgrid_pts = np.matmul(vgrid_pts, self.rotations[ind, ...])

        if return_spacing:
            return vgrid_pts, [x, y, z]
        return vgrid_pts


class ImplicitScene():
    def __init__(
            self,
            vgrid: VoxelGrid,
            voxel_size: float,
            latent_vecs: np.ndarray = None,
            sdf_decoder: torch.nn.Module = None,
            voxel_resolution: int = 8,
            device: torch.device = torch.device('cpu')
    ) -> None:
        self.vgrid = vgrid
        self.latent_vecs = latent_vecs
        self.decoder = sdf_decoder
        self.v_res = voxel_resolution
        self.device = device
        self.voxel_size = voxel_size

    def get_latent_vec(self, ind: int):
        return self.latent_vecs[ind, :]

    def trace_surface_points(
            self,
            z: np.ndarray,
            mask: np.ndarray = None,
            spacing: np.ndarray = None):
        if not isinstance(z, np.ndarray):
            z = z.detach().cpu().numpy()
        try:
            return marching_cubes(
                volume=z, level=0, mask=mask)
        except:
            return None

    def get_mesh_from_latent(
            self,
            latent_vec: np.ndarray,
            range: list = [-1, 1],
            canonical=False):
        self.decoder.eval()
        local_grid = self.vgrid.get_local_grid(
            v_res=self.v_res,
            range=range,
            canonical=False)
        local_grid = torch.from_numpy(local_grid).float().to(self.device)
        latent_vec = torch.from_numpy(latent_vec).float().to(self.device)
        latent_vec = latent_vec.unsqueeze(0).repeat(local_grid.shape[0], 1)
        x = torch.cat([latent_vec, local_grid], dim=-1)
        z = self.decoder(x)
        z = z.view(self.v_res, self.v_res, self.v_res)
        if torch.max(z) > 0 and torch.min(z) < 0:
            return self.trace_surface_points(z)
        else:
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('latents', type=str)
    parser.add_argument('cfg')
    parser.add_argument('ckpt')
    parser.add_argument('--orient', action='store_true')
    args = parser.parse_args()

    latent_vecs = np.load(args.latents)
    dataset = SampleDataset(args.data, args.orient, training=False)
    device = torch.device('cpu')
    network = ImplicitNet.create_from_cfg(args.cfg, args.ckpt, device)

    grid = VoxelGrid(
        dataset.voxels,
        dataset.voxel_size,
        dataset.centroids,
        dataset.rotations
    )

    scene = ImplicitScene(
        vgrid=grid,
        voxel_size=dataset.voxel_size,
        latent_vecs=latent_vecs,
        sdf_decoder=network,
        voxel_resolution=16,
        device=device
    )

    choices = np.random.permutation(grid.size())[:2]
    num_interp = 30
    num_per_row = 1
    while num_per_row*num_per_row < num_interp:
        num_per_row += 1

    start_latent = scene.get_latent_vec(choices[0])
    end_latent = scene.get_latent_vec(choices[1])
    latent_inc = (end_latent - start_latent)/num_interp

    geometries = []
    for i in range(num_interp):
        mesh = scene.get_mesh_from_latent(
            start_latent+latent_inc * i,
            canonical=False, range=[-.33, .33])
        if has(mesh):
            y = i // num_per_row
            x = i - num_per_row * y

            mesh = trimesh.Trimesh(mesh[0]*dataset.voxel_size, mesh[1])
            mesh.apply_translation(np.array([y*3,x*3, 0]))
            mesh = mesh.as_open3d
            mesh.compute_vertex_normals()
            geometries.append(mesh)
    o3d.visualization.draw_geometries(geometries)