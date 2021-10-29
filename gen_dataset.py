import argparse
import json
import os

import numpy as np
import open3d as o3d
import torch

from third_party.torchgp import (compute_sdf, load_obj, normalize,
                                 point_sample, sample_near_surface)


def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def display_sdf(pts, sdf):
    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud])


def get_voxels(cloud, voxel_size=0.1, method='uniform', voxel_num=1500):
    if method == 'uniform':
        voxels = torch.divide(cloud, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        return voxels
    elif method == 'random':
        voxels = torch.randperm(cloud.shape[0])[:voxel_num]
        voxels = cloud[voxels, :]
        return voxels


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


def cal_z_axis(ref_points: torch.Tensor):
    '''Calculate the z-axis from point clouds
    Args:
        ref_points(Tensor): needs to be centred
    '''
    cov_mat = torch.matmul(ref_points.transpose(-1, -2), ref_points)
    _, eig_vecs = torch.linalg.eigh(cov_mat)
    z_axis = eig_vecs[:, 0]
    mask = (torch.sum(-z_axis * ref_points, dim=1) < 0).float().unsqueeze(1)
    z_axis = z_axis * (1 - mask) - z_axis * mask
    return z_axis


def cal_x_axis(pnts, z_axis):
    z_proj = torch.dot(pnts, z_axis)
    sign_weight = z_proj**2
    sign_weight[z_proj < 0] *= -1

    vec_proj = torch.zeros((pnts.shape[0], 3))
    vec_proj = pnts - z_proj * z_axis

    dist = torch.norm(pnts, ord=2, axis=-1)
    supp = torch.max(dist)
    dist_weight = (supp - dist)**2
    x_axis = dist_weight[:, None] * sign_weight[:, None] * vec_proj
    x_axis = torch.sum(x_axis, axis=0)
    x_axis /= torch.norm(x_axis, ord=2)
    return x_axis


def cal_xyz_axis(surface, centroid):
    ref_pnts = surface - centroid
    z_axis = cal_z_axis(ref_pnts)
    x_axis = cal_x_axis(ref_pnts, z_axis)
    y_axis = torch.cross(x_axis, z_axis)
    return torch.stack([x_axis, y_axis, z_axis], dim=0)


# def cal_xyz_axis(surf, centroid):
#     pcd = surf - centroid
#     dist = np.linalg.norm(pcd, ord=2, axis=-1)
#     dist_ind = dist.argsort()
#     pcd = pcd[dist_ind, :]
#     dist = dist[dist_ind]

#     cov_mat = np.matmul(pcd.transpose(), pcd)
#     _, eig_vecs = np.linalg.eigh(cov_mat)
#     z_axis = eig_vecs[:, 0]
#     z_sign = 0.0
#     for i in range(pcd.shape[0]):
#         vec_x = 0-pcd[i, 0]
#         vec_y = 0-pcd[i, 1]
#         vec_z = 0-pcd[i, 2]
#         sign = (vec_x * z_axis[0] + vec_y * z_axis[1] + vec_z * z_axis[2])
#         z_sign += sign
#     if z_sign < 0:
#         z_axis *= -1
#     z_proj = np.dot(pcd, z_axis)
#     sign_weight = z_proj**2
#     sign_weight[z_proj < 0] *= -1

#     vec_proj = np.zeros((pcd.shape[0], 3))
#     for i in range(pcd.shape[0]):
#         vec_proj[i, 0] = pcd[i, 0] - z_proj[i] * z_axis[0]
#         vec_proj[i, 1] = pcd[i, 1] - z_proj[i] * z_axis[1]
#         vec_proj[i, 2] = pcd[i, 2] - z_proj[i] * z_axis[2]

#     supp = np.max(dist)
#     dist_weight = (supp - dist)**2
#     x_axis = dist_weight[:, None] * sign_weight[:, None] * vec_proj
#     x_axis = np.sum(x_axis, axis=0)
#     x_axis /= np.linalg.norm(x_axis, ord=2)

#     y_axis = np.cross(z_axis, x_axis)
#     rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
#     return rotation


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
        centroid = torch.mean(pnts, dim=0)

        rotation = torch.eye(3)
        if pnts.shape[0] > 10:
            rotations = cal_xyz_axis(
                pnts, centroid)
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
        'examples/shapenet/sv2_chairs_train.json',
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
                    surface_pnts.cuda(), samples.cuda(),
                    voxels.cuda(), voxel_size)

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
