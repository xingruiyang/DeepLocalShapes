import argparse
import json
import os

import torch

from match import LatentMatcher
from network import ImplicitNet
from utils import load_model
import trimesh
import pickle
import numpy as np


def normalize_latents(latents):
    norm = np.linalg.norm(latents, ord=2, axis=-1, keepdims=True)
    norm[norm == 0] == 1
    return latents / norm


def load_data(data_path, misc_path, load_orient=False, load_mesh=False, prefix='src'):
    input_data = dict()
    data = np.load(os.path.join(data_path, 'meta.npz'))
    input_data[prefix+'_voxels'] = data['voxels']
    input_data[prefix + '_latents'] = np.load(os.path.join(misc_path, 'latent_vecs.npy'))
    input_data[prefix + '_latents'] = normalize_latents(input_data[prefix + '_latents'])
    input_data['voxel_size'] = data['voxel_size']

    if load_orient:
        input_data['rotations'] = data['rotations']
        input_data['centroids'] = data['centroids']

    if load_mesh:
        input_data['query_pts'] = trimesh.load(
            os.path.join(misc_path, 'recon.ply')).sample(50000)

    return input_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('latents', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--num-iter', type=int, default=10)
    parser.add_argument('--dist-th', type=float, default=0.1)
    parser.add_argument('--network-cfg', type=str, default=None)
    parser.add_argument('--network-ckpt', type=str, default=None)
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--icp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda:0' if (
        (not args.cpu) and torch.cuda.is_available()) else 'cpu')

    match_args = dict()
    match_args['distance_threshold'] = args.dist_th
    if args.icp and args.network_cfg is not None:
        network_args = json.load(open(args.network_cfg, 'r'))
        network = ImplicitNet(**network_args['params'])
        load_model(args.network_ckpt, network)
        match_args['network'] = network

    splits = {
        "7-scenes-redkitchen": 60,
        # "sun3d-mit_76_studyroom-76-1studyroom2": 66,
        # "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika": 38,
        "sun3d-home_at-home_at_scan1_2013_jan_1": 60,
        # "sun3d-home_md-home_md_scan9_2012_sep_30": 60,
        # "sun3d-hotel_uc-scan3": 55,
        # "sun3d-hotel_umd-maryland_hotel1": 57,
        "sun3d-hotel_umd-maryland_hotel3": 36
    }

    for scene_name, num_frags in splits.items():
        print("processing {}".format(scene_name))
        indices = []
        poses = []
        data_path = os.path.join(args.data, scene_name)
        misc_path = os.path.join(args.latents, scene_name)
        for frag_idx in range(num_frags):
            src_data = load_data(
                os.path.join(data_path, str(frag_idx)),
                os.path.join(misc_path, str(frag_idx)),
                False, args.icp)
            for frag_idx2 in range(frag_idx+2, num_frags):
                print("computing src {} and dst {}".format(frag_idx, frag_idx2))
                dst_data = load_data(
                    os.path.join(data_path, str(frag_idx2)),
                    os.path.join(misc_path, str(frag_idx2)),
                    True, False, 'dst')
                input_data = {**match_args, **src_data, **dst_data}
                matcher = LatentMatcher(**input_data)
                result = matcher.compute_rigid_transform()
                transform = result.transformation

                if args.icp:
                    transform = matcher.refine_pose(
                        transform, args.num_iter, True)

                poses.append(np.linalg.inv(transform))
                indices.append("{} {} {}\n".format(
                    frag_idx, frag_idx2, num_frags))

        out_path = os.path.join(
            args.output, scene_name)
        os.makedirs(out_path, exist_ok=True)
        log_file = os.path.join(out_path, "pred.log")
        with open(log_file, "w") as f:
            for i in range(len(indices)):
                f.write(indices[i])
                pose = poses[i]
                for j in range(4):
                    f.write("{} {} {} {}\n".format(
                        pose[j, 0], pose[j, 1],
                        pose[j, 2], pose[j, 3]))
