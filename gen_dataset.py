import argparse
import json
import os

import numpy as np
import open3d as o3d
import torch

from third_party.torchgp import (compute_sdf, load_obj, normalize,
                                 point_sample, sample_near_surface)
from utils import cal_xyz_axis


def get_voxels(pnts: torch.Tensor,
               voxel_size: float = 0.1,
               method: str = 'uniform',
               voxel_num: int = 1500):
    """Compute voxels from the given point cloud
    Args:
        pnts (tensor): pnts of shape Nx3
        voxels (float, optional): the length of voxels
        method (str, optional): sampling strategy
        voxel_num (int, optional): only used when `method` is set to `random`
    """
    if method == 'uniform':
        voxels = torch.divide(pnts, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        return voxels
    elif method == 'random':
        voxels = torch.randperm(pnts.shape[0])[:voxel_num]
        voxels = pnts[voxels, :]
        return voxels


def sample_voxels(centres: torch.Tensor, num_samples: int, range: float):
    '''Sample uniformly in [-range,range] bounding volume within voxels
    Args:
        centres (tensor)  : set of centres to sample from
        num_samples (int) : number of points to sample
        range (float): range to sample points
    '''
    samples = torch.rand(
        centres.shape[0], num_samples, 3, device=centres.device)
    samples = samples * 2.0 * range - range
    samples = centres[..., None, :3] + samples
    return samples.reshape(-1, 3)


def get_samples(surface, samples, voxels, voxel_size):
    data = []
    num_voxels = voxels.shape[0]

    centroids = []
    rotations = []
    for i in range(num_voxels):
        voxel = voxels[i, :]
        pnts = samples[:, :3] - voxel
        selector = torch.norm(pnts, p=2, dim=-1)
        selector = selector < (1.5 * voxel_size)
        pnts = pnts[selector, :] / (1.5 * voxel_size)
        sdf = samples[selector, 3] / (1.5 * voxel_size)
        indices = torch.from_numpy(np.asarray([i]*pnts.shape[0]))
        data.append(torch.cat(
            [indices[:, None].cuda(), pnts, sdf[:, None]], dim=-1))

        pnts = surface - voxel
        selector = torch.norm(pnts, p=2, dim=-1)
        selector = selector < (1.5 * voxel_size)
        pnts = pnts[selector, :] / (1.5 * voxel_size)

        rotation = torch.eye(3)
        centroid = torch.mean(pnts, dim=0)
        if pnts.shape[0] > 10:
            ref_pnts = pnts - centroid
            rotation = cal_xyz_axis(ref_pnts)
        rotations.append(rotation)
        centroids.append(centroid)

    return torch.cat(data, dim=0).detach().cpu().numpy(), \
        torch.stack(rotations, axis=0).detach().cpu().numpy(), \
        torch.stack(centroids, axis=0).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--nshape_cat', type=int, default=50)
    parser.add_argument('--save_iterm', action='store_true')
    args = parser.parse_args()

    split_filenames = [
        'splits/shapenet/sv2_chairs_train.json',
        'splits/shapenet/sv2_lamps_train.json',
        'splits/shapenet/sv2_planes_train.json',
        'splits/shapenet/sv2_sofas_train.json',
        'splits/shapenet/sv2_tables_train.json'
    ]

    for split_file in split_filenames:
        object_list = json.load(open(split_file, 'r'))['ShapeNetV2']
        for key, value in object_list.items():
            cate_out = os.path.join(args.out_path, key)
            os.makedirs(cate_out, exist_ok=True)
            num_sampled = 0
            for filename in value:
                print("processing {}th model {}".format(num_sampled, filename))
            # try:
                object_path = os.path.join(
                    args.data_path, key, filename,  'models/model_normalized.obj')
                verts, faces = load_obj(object_path)
                verts, faces = normalize(verts, faces)
                voxel_size = torch.max(torch.norm(verts, p=2, dim=-1)) / 32
                surface_pnts = point_sample(
                    verts, faces, ['trace'], 200000)
                voxels = get_voxels(
                    surface_pnts, voxel_size, method='uniform')

                surface_samples = []
                surface_samples.append(sample_near_surface(
                    verts, faces, 200000, 0.0025))
                surface_samples.append(sample_near_surface(
                    verts, faces, 200000, 0.0005))
                surface_samples.append(sample_voxels(
                    voxels, 32, 1.5 * voxel_size))

                surface_samples = torch.cat(surface_samples, dim=0)
                sdf = compute_sdf(
                    verts.cuda(), faces.cuda(), surface_samples.cuda())
                samples = torch.cat(
                    [surface_samples, sdf.cpu()[:, None]], dim=-1)

                if args.save_iterm:
                    obj_out = os.path.join(
                        cate_out, "{}_interm.npz".format(filename))
                    np.savez(obj_out,
                             surface=surface_pnts.cpu().numpy(),
                             samples=samples.cpu().numpy(),
                             voxels=voxels.cpu().numpy(),
                             voxel_size=voxel_size.item())

                samples, rotations, centroids = get_samples(
                    surface_pnts.cuda(), samples.cuda(), voxels.cuda(), voxel_size)

                sample_outpath = os.path.join(
                    cate_out, "{}.npy".format(filename))
                # np.savez(sample_outpath,
                #          samples=samples,
                #          rotaions=rotations,
                #          centroids=centroids)

            # except Exception:
            #     print("failed at loading model {}".format(filename))

                num_sampled += 1
                if num_sampled >= args.nshape_cat:
                    break
