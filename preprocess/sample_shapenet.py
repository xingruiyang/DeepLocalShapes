import argparse
import json
import os
import pickle

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from mesh_sampler import MeshSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('splits', type=str)
    parser.add_argument('--num-samples', type=int, default=20)
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

    splits = json.load(open(args.splits, 'r'))['ShapeNetV2']
    for key, value in splits.items():
        for shape in value:
            print('sampling {}'.format(shape))
            input_dir = os.path.join(args.in_dir, key, shape)
            output_dir = os.path.join(args.out_dir, key, shape)

            if os.path.exists(output_dir):
                continue

            model_path = os.path.join(
                input_dir, 'models/model_normalized.obj')
            mesh = trimesh.load(model_path)

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

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "samples.pkl"), "wb") as f:
                pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(output_dir, sample_name),
                    samples.astype(np.float32))
            np.save(os.path.join(output_dir, surface_pts_name),
                    surface.astype(np.float32))
            np.save(os.path.join(output_dir, surface_sdf_name),
                    surface_sdf.astype(np.float32))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(True)
            mesh.export(os.path.join(output_dir, "gt.ply"))
