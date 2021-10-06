import argparse
import os
import pickle

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from mesh_sampler import MeshSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--pts-per-voxel', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--use-depth', action='store_true')
    parser.add_argument('--random-rot', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    sampler = MeshSampler(
        args.voxel_size,
        args.pts_per_voxel,
        args.network,
        args.normalize,
        args.use_depth)

    mesh = trimesh.load(args.mesh)

    samples = sampler.sample_sdf(mesh, return_surface=True)
    samples, voxels, centroids, rotations, \
        surface, rand_surface, rand_sdf = samples
    surface_sdf = np.concatenate(
        [rand_surface, rand_sdf[:, None]], axis=-1)

    sample_name = 'samples.npy'
    surface_pts_name = 'surface_pts.npy'
    surface_sdf_name = 'surface_sdf.npy'

    out = dict()
    out['samples'] = sample_name
    out['surface_pts'] = surface_pts_name
    out['surface_sdf'] = surface_sdf_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
    np.save(os.path.join(args.output, surface_pts_name),
            surface.astype(np.float32))
    np.save(os.path.join(args.output, surface_sdf_name),
            surface_sdf.astype(np.float32))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(True)
    mesh.export(os.path.join(args.output, "gt.ply"))