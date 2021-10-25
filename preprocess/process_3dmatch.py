import argparse
import glob
import os
from itertools import chain

import natsort
import numpy as np
import open3d as o3d
import torch
from sklearn.neighbors import KDTree

from depth_sampler import DepthSampler

scene_list = [
    '7-scenes-redkitchen',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3'
]


def toldi_compute_xyz(surf, centroid):
    pcd = surf - centroid
    dist = np.linalg.norm(pcd, ord=2, axis=-1)
    dist_ind = dist.argsort()
    pcd = pcd[dist_ind, :]
    dist = dist[dist_ind]

    cov_mat = np.matmul(pcd.transpose(), pcd)
    _, eig_vecs = np.linalg.eigh(cov_mat)
    z_axis = eig_vecs[:, 0]
    z_sign = 0.0
    for i in range(pcd.shape[0]):
        vec_x = 0-pcd[i, 0]
        vec_y = 0-pcd[i, 1]
        vec_z = 0-pcd[i, 2]
        sign = (vec_x * z_axis[0] + vec_y * z_axis[1] + vec_z * z_axis[2])
        z_sign += sign
    if z_sign < 0:
        z_axis *= -1
    z_proj = np.dot(pcd, z_axis)
    sign_weight = z_proj**2
    sign_weight[z_proj < 0] *= -1

    vec_proj = np.zeros((pcd.shape[0], 3))
    for i in range(pcd.shape[0]):
        vec_proj[i, 0] = pcd[i, 0] - z_proj[i] * z_axis[0]
        vec_proj[i, 1] = pcd[i, 1] - z_proj[i] * z_axis[1]
        vec_proj[i, 2] = pcd[i, 2] - z_proj[i] * z_axis[2]

    supp = np.max(dist)
    dist_weight = (supp - dist)**2
    x_axis = dist_weight[:, None] * sign_weight[:, None] * vec_proj
    x_axis = np.sum(x_axis, axis=0)
    x_axis /= np.linalg.norm(x_axis, ord=2)

    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    return rotation


def to_o3d(arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


def get_voxels(cloud, voxel_size=0.1, method='uniform', voxel_num=1500):
    if method == 'uniform':
        voxels = cloud // voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= voxel_size
        return voxels
    elif method == 'random':
        voxels = cloud // voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels = np.random.permutation(cloud.shape[0])[:voxels.shape[0]*3//2]
        voxels = cloud[voxels, :]
        return voxels
    elif method == 'closest':
        voxels = cloud // voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= voxel_size

        kd_tree = KDTree(cloud)
        dist, ind = kd_tree.query(voxels)
        voxels = cloud[ind[:, 0], :]
        return voxels


def draw_voxels(cloud, voxels, voxel_size=0.1):
    num_voxels = voxels.shape[0]
    geometries = []
    point_cloud = to_o3d(cloud)
    for i in range(num_voxels):
        voxel = voxels[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        geometries.append(bbox_inner)
    o3d.visualization.draw_geometries(geometries+[point_cloud])


def get_frame_selector(scene_name, num_imgs, frames_per_frag=50):
    if scene_name == 'sun3d-hotel_uc-scan3':
        return chain(range(0, 2750-1, 100), range(7500, num_imgs-frames_per_frag-1, 100))
    elif scene_name == 'sun3d-home_at-home_at_scan1_2013_jan_1' or scene_name == 'sun3d-home_md-home_md_scan9_2012_sep_30':
        return range(0, 6000-frames_per_frag-1, 100)
    elif scene_name == 'sun3d-hotel_umd-maryland_hotel1':
        return range(0, num_imgs-frames_per_frag-1, 100)
    elif scene_name == '7-scenes-redkitchen':
        return range(0, num_imgs, frames_per_frag)
    else:
        return range(0, num_imgs-frames_per_frag-1, frames_per_frag)


def generate_point_cloud(args, scene_list):
    for scene_name in scene_list:
        scene_path = os.path.join(args.path, scene_name)
        print("processing {}".format(scene_path))
        seq_list = [f for f in os.listdir(
            scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        seq_list = natsort.natsorted(seq_list)
        intr_path = os.path.join(scene_path, 'camera-intrinsics.txt')
        intr = np.loadtxt(intr_path)

        frag_idx = 0
        for seq_id in range(min(len(seq_list), 3)):
            print("processing seq {}".format(seq_list[seq_id]))
            seq_name = seq_list[seq_id]
            depth_imgs = glob.glob(os.path.join(
                args.path, scene_name, seq_name, "*.depth.png"))
            depth_imgs = natsort.natsorted(depth_imgs)
            frame_selector = get_frame_selector(
                scene_name, len(depth_imgs), args.frames_per_frag)
            for ind in frame_selector:
                print("processing frag {}".format(frag_idx))
                sampler = DepthSampler(
                    scene_path,
                    seq_name,
                    False,
                    args.skip_frames,
                    args.depth_limit,
                    frame_selector=range(
                        ind,
                        ind + args.frames_per_frag,
                        args.skip_frames)
                )

                point_cloud = sampler.sample_sdf()
                print(point_cloud.shape)
                out_path = os.path.join(args.output, scene_name, str(frag_idx))
                os.makedirs(out_path, exist_ok=True)
                np.save(os.path.join(out_path, 'cloud.npy'), point_cloud)


def generate_voxels(args):
    for scene_name in scene_list:
        scene_path = os.path.join(args.path, scene_name)
        print("processing {}".format(scene_path))
        frag_list = [f for f in os.listdir(
            scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        frag_list = natsort.natsorted(frag_list)
        for frag_id in range(len(frag_list)):
            print("processing fragment {}".format(frag_id))
            frag_filename = os.path.join(
                args.path, scene_name, str(frag_id), 'cloud.npy')
            cloud = np.load(frag_filename)
            points = cloud[:, 1:4]
            voxels = get_voxels(points, args.voxel_size, method='random')
            # draw_voxels(points, voxels)
            num_voxels = voxels.shape[0]
            rotations = []
            centroids = []
            points = torch.from_numpy(points).cuda()
            for i in range(num_voxels):
                print('{}/{}'.format(i, num_voxels))
                voxel = voxels[i, ...]
                voxel_pts = points - torch.from_numpy(voxel).cuda()
                dist = torch.norm(voxel_pts, p=2, dim=-1)
                voxel_pts = voxel_pts[dist < (1.5*args.voxel_size), :]
                voxel_pts = voxel_pts.detach().cpu().numpy()
                centroid = np.mean(voxel_pts, axis=0)
                rotation = np.eye(3)
                if voxel_pts.shape[0] > 2048:
                    voxel_pts = voxel_pts[np.random.permutation(
                        voxel_pts.shape[0])[:2048], :]
                if voxel_pts.shape[0] > 10:
                    rotation = toldi_compute_xyz(voxel_pts, centroid)
                rotations.append(rotation)
                centroids.append(centroid)

            centroids = np.stack(centroids, axis=0)
            rotations = np.stack(rotations, axis=0)
            out_path = os.path.join(args.path, scene_name, str(frag_id))
            np.savez(os.path.join(out_path, 'meta.npz'),
                     voxels=voxels, centroids=centroids,
                     rotations=rotations, voxel_size=args.voxel_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--frames-per-frag', type=int, default=50)
    parser.add_argument('--skip-frames', type=int, default=10)
    parser.add_argument('--depth-limit', type=float, default=10)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--mnfld-pnts', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()

    generate_voxels(args)
