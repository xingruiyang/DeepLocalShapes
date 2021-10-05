import argparse
import os
import pickle
from random import shuffle

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from mesh_sampler import MeshSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--pts-per-voxel', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--use-depth', action='store_true')
    parser.add_argument('--random-rot', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    cate_dir = {
        # 'sofa': '04256520',
        'airliner': '02691156',
        # 'lamp': '03636649',
        # 'chair': '03001627',
        # 'table': '04379243'
    }

    sampler = MeshSampler(
        args.voxel_size,
        args.pts_per_voxel,
        args.network,
        args.normalize,
        args.use_depth)

    for key, value in cate_dir.items():
        input_dir_cat = os.path.join(args.in_dir, value)
        output_dir_cat = os.path.join(args.out_dir, key)
        instance_models = [f for f in os.listdir(
            input_dir_cat) if os.path.isdir(os.path.join(input_dir_cat, f))]
        shuffle(instance_models)
        num_total_models = len(instance_models)

        num_samples = 0
        for ind, model in enumerate(instance_models):
            model_path = os.path.join(
                input_dir_cat, model, 'models/model_normalized.obj')
            model_path_out = os.path.join(
                output_dir_cat, '{}/'.format(num_samples))
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

            os.makedirs(model_path_out, exist_ok=True)
            with open(os.path.join(model_path_out, "samples.pkl"), "wb") as f:
                pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(model_path_out, sample_name),
                    samples.astype(np.float32))
            np.save(os.path.join(model_path_out, surface_pts_name),
                    surface.astype(np.float32))
            np.save(os.path.join(model_path_out, surface_sdf_name),
                    surface_sdf.astype(np.float32))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(True)
            mesh.export(os.path.join(model_path_out, "gt.ply"))

            num_samples += 1
            if num_samples >= args.num_samples:
                break
