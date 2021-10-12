import random
import glob
import argparse
import os
import pickle

import numpy as np
import torch
import trimesh


class Voxelizer(object):
    def __init__(self, pcd_path) -> None:
        super().__init__()
        samples = np.load(pcd_path+'.npy')
        self.points = samples[:, :3]
        self.sdf = samples[:, 3]
        self.surface = np.load(pcd_path+'_surf.npy')

    def display_sdf(self, pts, sdf):
        color = np.zeros_like(pts)
        color[sdf > 0, 0] = 1
        color[sdf < 0, 2] = 1
        trimesh.PointCloud(pts, color).show()

    def create_voxels(self, voxel_size):
        points = torch.from_numpy(self.points).cuda()
        sdfs = torch.from_numpy(self.sdf).cuda()
        surface = torch.from_numpy(self.surface).cuda()

        voxels = torch.div(surface, voxel_size, rounding_mode='floor')
        voxels = torch.unique(voxels, dim=0)
        voxels += .5
        voxels *= voxel_size
        print("{} voxels to be sampled".format(voxels.shape[0]))

        max_num_pcd = 2**14
        samples = []
        centroids = []
        rotations = []
        surf_pts = []
        for vid in range(voxels.shape[0]):
            # print("{}/{}".format(vid, voxels.shape[0]))
            voxel = voxels[vid, :]
            pcd = points - voxel
            dist = torch.norm(pcd, p=np.inf, dim=-1)
            selector = dist < (1.5 * voxel_size)
            pcd = pcd[selector, :]
            sdf = sdfs[selector]

            if pcd.shape[0] > max_num_pcd:
                rand_sel = torch.randperm(pcd.shape[0])[:max_num_pcd]
                pcd = pcd[rand_sel, :]
                sdf = sdf[rand_sel]

            surf = surface - voxel
            dist = torch.norm(surf, p=np.inf, dim=-1)
            selector = dist < (1.5 * voxel_size)
            surf = surf[selector, :]
            if surf.shape[0] >= max_num_pcd:
                rand_sel = torch.randperm(pcd.shape[0])[:max_num_pcd]
                surf = surf[rand_sel, :]

            # global_pts = (torch.randn((256, 3)) * 3 - 1.5) * voxel_size
            # global_sdf = torch.ones((256, )) * -1
            # global_weight = torch.zeros((256, ))

            sample_pts = torch.cat([
                pcd,
                # global_pts.cuda()
            ], axis=0)
            sample_sdf = torch.cat([
                sdf,
                # global_sdf.cuda()
            ], axis=0)
            sample_weights = torch.cat([
                torch.ones_like(sdf),
                # global_weight.cuda()
            ], axis=0)

            # self.display_sdf(sample_pts.detach().cpu(),
            #                  sample_sdf.detach().cpu())

            sample = np.zeros((sample_pts.shape[0], 6))
            sample[:, 0] = vid
            sample[:, 1:4] = sample_pts.detach().cpu().numpy()
            sample[:, 4] = sample_sdf.detach().cpu().numpy()
            sample[:, 5] = sample_weights.detach().cpu().numpy()
            samples.append(sample)
            surf_pts.append(surf.detach().cpu().numpy())
            centroids.append(np.zeros(3,))
            rotations.append(np.eye(3))

        samples = np.concatenate(samples, axis=0)
        surf_pts = np.concatenate(surf_pts, axis=0)
        centroids = np.stack(centroids, axis=0)
        rotations = np.stack(rotations, axis=0)
        voxels = voxels.detach().cpu().numpy()

        print("{} points sampled with {} voxels aligned".format(
            samples.shape[0], rotations.shape[0]))

        return samples, voxels, centroids, rotations, surf_pts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--num-shapes', type=int, default=50)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    args = parser.parse_args()

    subdirs = glob.glob(os.path.join(args.in_dir, "*_surf.npy"))

    all_samples = []
    all_voxels = []
    all_surface = []
    all_centroids = []
    all_rotations = []

    random.shuffle(subdirs)
    num_total_voxels = 0
    num_shape_sampled = 0
    for index, filename in enumerate(subdirs):
        print(index)

        voxelizer = Voxelizer(filename[:-9])
        samples = voxelizer.create_voxels(args.voxel_size)
        samples, voxels, centroids, rotations, surface = samples

        if np.isnan(samples).any():
            print("has nan!")
            continue

        y = num_shape_sampled // 10
        x = num_shape_sampled - y * 10

        samples[:, 0] += num_total_voxels
        all_samples.append(samples)
        all_voxels.append(voxels+np.array([x, y, 0])*1.5)
        all_surface.append(surface)
        all_centroids.append(centroids)
        all_rotations.append(rotations)
        num_total_voxels += voxels.shape[0]

        num_shape_sampled += 1
        if num_shape_sampled > args.num_shapes:
            break

    all_voxels = np.concatenate(all_voxels, axis=0)
    # all_surface = np.concatenate(all_surface, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    all_centroids = np.concatenate(all_centroids, axis=0)
    all_rotations = np.concatenate(all_rotations, axis=0)

    sample_name = 'samples.npy'
    surface_pts_name = 'surface_pts.npy'
    surface_sdf_name = 'surface_sdf.npy'

    out = dict()
    out['samples'] = sample_name
    out['surface_pts'] = surface_pts_name
    out['surface_sdf'] = surface_sdf_name
    out['voxels'] = all_voxels.astype(np.float32)
    out['centroids'] = all_centroids.astype(np.float32)
    out['rotations'] = all_rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            all_samples.astype(np.float32))
    # np.save(os.path.join(args.output, surface_pts_name),
    #         all_surface.astype(np.float32))
