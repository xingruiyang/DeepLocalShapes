import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

from network import PointNetTransformer
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


def save_samples(surface: torch.Tensor,
                 samples: torch.Tensor,
                 voxels: torch.Tensor,
                 voxel_size: float,
                 ckpt: str,
                 out_path: str):
    '''Split sample pnts into voxel grids and save as npz format
    Args:
        surface (tensor)  : surface pnts to esitmate transformations
        samples (tensor) : a set of pnts-sdf pairs to sample from
        voxels (tensor): a set of voxels to sample from
        voxel_size (float): the length of voxels
        ckpt (str): checkpoints to the transformer network
        out_path (str): output directory
    '''
    #transformer = PointNetTransformer.create_from_ckpt(ckpt)
    # transformer.eval().cuda()
    data = []
    centroids = []
    rotations = []

    num_voxels = voxels.shape[0]
    for i in range(num_voxels):
        voxel = voxels[i, :]
        pnts = samples[:, :3] - voxel
        selector = torch.norm(pnts, p=2, dim=-1)
        selector = selector < (1.5 * voxel_size)
        pnts = pnts[selector, :] / (1.5 * voxel_size)
        sdf = samples[selector, 3] / (1.5 * voxel_size)
        indices = torch.from_numpy(np.asarray([i]*pnts.shape[0]))

        surface_pnts = surface - voxel
        selector = torch.norm(surface_pnts, p=2, dim=-1)
        selector = selector < (1.5 * voxel_size)
        surface_pnts = surface_pnts[selector, :] / (1.5 * voxel_size)

        rotation = torch.eye(3)
        centroid = torch.mean(pnts, dim=0)
        if surface_pnts.shape[0] > 10:
            ref_pnts = surface_pnts - centroid
            rotation = cal_xyz_axis(ref_pnts)
            # rotation = transformer(
            #     ref_pnts[None, ...],
            #     transpose_input=True)[0, ...].reshape(3, 3)
            # check rotation matrices
            # scene = trimesh.Scene()
            # for i in range(3):
            #     for j in range(3):
            #         ref_pnts2 = deepcopy(ref_pnts)
            #         ref_pnts2 += torch.randn_like(ref_pnts2) * 0.05
            #         ref_pnts2 = torch.matmul(
            #             ref_pnts2, torch.from_numpy(R.random().as_matrix()).cuda().float())
            #         # rotation = transformer(
            #         #     ref_pnts2[None, ...],
            #         #     transpose_input=True)[0, ...].reshape(3, 3)
            #         rotation = cal_xyz_axis(ref_pnts2)
            #         ref_pnts2 = torch.matmul(
            #             ref_pnts2, rotation.transpose(0, 1))
            #         transform = np.eye(4)
            #         transform[0, 3] = i * 2
            #         transform[1, 3] = j * 2
            #         scene.add_geometry(trimesh.PointCloud(
            #             ref_pnts2.detach().cpu()), transform=transform)
            # scene.show()
        data.append(torch.cat(
            [indices[:, None].cuda(), pnts, sdf[:, None]],
            dim=-1).detach().cpu().numpy())
        rotations.append(rotation.detach().cpu().numpy())
        centroids.append(centroid.detach().cpu().numpy())

    voxels = voxels.detach().cpu().numpy()
    data = np.concatenate(data, axis=0)
    rotations = np.stack(rotations, axis=0)
    centroids = np.stack(centroids, axis=0)
    np.savez(out_path,
             samples=data,
             voxels=voxels,
             voxel_size=voxel_size,
             rotations=rotations,
             centroids=centroids)


if __name__ == '__main__':
    '''Sample training/evaluation examples from the given ShapeNet models
    You need to provide path to the ShapeNetCoreV.2 dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--nshape_cat', type=int, default=50)
    parser.add_argument('--save_iterm', action='store_true')
    args = parser.parse_args()

    # split_filenames = [
    #     'splits/shapenet/sv2_chairs_train.json',
    #     'splits/shapenet/sv2_lamps_train.json',
    #     'splits/shapenet/sv2_planes_train.json',
    #     'splits/shapenet/sv2_sofas_train.json',
    #     'splits/shapenet/sv2_tables_train.json'
    # ]

    split_filenames = [
        'splits/shapenet/sv2_chairs_test.json',
        'splits/shapenet/sv2_lamps_test.json',
        'splits/shapenet/sv2_planes_test.json',
        'splits/shapenet/sv2_sofas_test.json',
        'splits/shapenet/sv2_tables_test.json'
    ]

    for split_file in split_filenames:
        object_list = json.load(open(split_file, 'r'))['ShapeNetV2']
        for key, value in object_list.items():
            cate_out = os.path.join(args.out_path, key)
            os.makedirs(cate_out, exist_ok=True)
            num_sampled = 0
            for filename in value:
                print("processing {}th model {}".format(num_sampled, filename))
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

                sample_outpath = os.path.join(
                    cate_out, "{}.npz".format(filename))
                save_samples(surface_pnts.cuda(),
                             samples.cuda(),
                             voxels.cuda(),
                             voxel_size,
                             args.ckpt,
                             sample_outpath)

                num_sampled += 1
                if num_sampled >= args.nshape_cat:
                    break
