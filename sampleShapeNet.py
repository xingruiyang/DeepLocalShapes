import argparse
import json
import os
import pickle

import numpy as np
import open3d as o3d
import torch
import trimesh

from third_party.torchgp import (compute_sdf, load_obj, normalize,
                                 point_sample, sample_near_surface)


def display_sdf(pts, sdf):
    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    trimesh.PointCloud(pts, color).show()


def get_voxels(cloud, voxel_size=0.1, method='uniform', voxel_num=1500):
    if method == 'uniform':
        voxels = torch.divide(cloud, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        return voxels
    elif method == 'random':
        voxels = np.random.permutation(cloud.shape[0])[:voxel_num]
        voxels = cloud[voxels, :]
        return voxels


def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def draw_voxels(pts, voxels=None, voxel_size=0.1):
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()
    if voxels is None:
        voxels = get_voxels(pts, voxel_size, method='uniform')
    num_voxels = voxels.shape[0]
    print("num voxels: ", num_voxels)
    geometries = []
    point_cloud = to_o3d(pts)
    for i in range(num_voxels):
        voxel = voxels[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        geometries.append(bbox_inner)
    o3d.visualization.draw_geometries(geometries+[point_cloud])


def sample_voxels(
        centres: torch.Tensor,
        num_samples: int,
        range: float):
    """Sample uniformly in [-range,range] bounding volume within voxels

    Args:
        centres (tensor)  : set of centres to sample from
        num_samples (int) : number of points to sample
    """
    samples = torch.rand(
        centres.shape[0], num_samples, 3, device=centres.device)
    samples = samples * 2.0 * range - range
    samples = centres[..., None, :3] + samples
    return samples.reshape(-1, 3)


def split_data(samples, voxels, voxel_size):
    data = []
    for i in range(voxels.shape[0]):
        voxel = voxels[i, :]
        pnts = samples[:, :3] - voxel
        selector = np.linalg.norm(pnts, ord=2, axis=-1)
        selector = selector < (1.5 * voxel_size)
        pnts = pnts[selector, :] / (1.5*voxel_size)
        sdf = samples[selector, 3] / (1.5*voxel_size)
        data.append(np.concatenate([pnts, sdf], axis=-1))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    split_filenames = [
        # 'examples/shapenet/sv2_chairs_train.json',
        'examples/shapenet/sv2_lamps_train.json',
        'examples/shapenet/sv2_planes_train.json',
        'examples/shapenet/sv2_sofas_train.json',
        'examples/shapenet/sv2_tables_train.json'
    ]

    for split_file in split_filenames:
        object_list = json.load(open(split_file, 'r'))['ShapeNetV2']
        for key, value in object_list.items():
            cate_out = os.path.join(args.out_path, key)
            os.makedirs(cate_out, exist_ok=True)
            num_sampled = 0
            for filename in value:
                print("processing {}th model {}".format(num_sampled, filename))
                try:
                    object_path = os.path.join(
                        args.data_path, key, filename,  'models/model_normalized.obj')
                    verts, faces = load_obj(object_path)
                    verts, faces = normalize(verts, faces)
                    voxel_size = torch.max(torch.norm(verts, p=2, dim=-1)) / 32
                    print(voxel_size)
                    surface_pnts = point_sample(
                        verts, faces, ['trace'], 200000)
                    voxels = get_voxels(
                        surface_pnts, voxel_size, method='uniform')

                    surface_samples = []
                    surface_samples.append(
                        sample_near_surface(verts, faces, 200000, 0.0025))
                    surface_samples.append(
                        sample_near_surface(verts, faces, 200000, 0.0005))
                    surface_samples.append(
                        sample_voxels(voxels, 32, 1.5 * voxel_size))

                    surface_samples = torch.cat(surface_samples, dim=0)
                    sdf = compute_sdf(verts.cuda(), faces.cuda(),
                                      surface_samples.cuda())

                    samples = torch.cat(
                        [surface_samples, sdf.cpu()[:, None]], dim=-1)

                    obj_out = os.path.join(cate_out, "{}.npz".format(filename))
                    np.savez(obj_out,
                             samples=samples.cpu().numpy(),
                             voxels=voxels.cpu().numpy(),
                             voxel_size=voxel_size.item())
                    # surface_samples = surface_samples[sdf < 0, :]
                    # sdf = sdf[sdf < 0]
                    # display_sdf(surface_samples.cpu(), sdf.cpu())
                    # draw_voxels(surface_samples, voxels, voxel_size.item())
                except Exception:
                    print("failed at loading model {}".format(filename))

                num_sampled += 1
                if num_sampled >= 50:
                    break
