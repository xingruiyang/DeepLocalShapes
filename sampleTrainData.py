import argparse
import numpy as np
import json
import os
import torch


def get_samples(samples, voxels, voxel_size):
    data = []
    num_voxels = voxels.shape[0]

    samples = torch.from_numpy(samples).cuda()
    voxels = torch.from_numpy(voxels).cuda()

    for i in range(num_voxels):
        voxel = voxels[i, :]
        pnts = samples[:, :3] - voxel
        selector = torch.norm(pnts, p=2, dim=-1)
        selector = selector < (1.5 * voxel_size)
        pnts = pnts[selector, :] / (1.5 * voxel_size)
        sdf = samples[selector, 3] / (1.5 * voxel_size)
        indices = torch.from_numpy(np.asarray([i]*pnts.shape[0])).cuda()
        data.append(torch.cat(
            [indices[:, None], pnts, sdf[:, None]], dim=-1))
    return torch.cat(data, axis=0).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    categories = os.listdir(args.data_path)
    for cat in categories:
        cat_path = os.path.join(args.data_path, cat)
        cat_out = os.path.join(args.out_path, cat)
        os.makedirs(cat_out, exist_ok=True)
        obj_files = os.listdir(cat_path)
        for filename in obj_files:
            filepath = os.path.join(cat_path, filename)
            out_path = os.path.join(
                cat_out, '{}.npy'.format(filename.split('.')[0]))
            print("processing model {}".format(filename))

            data_point = np.load(filepath)
            samples = data_point['samples']
            voxels = data_point['voxels']
            voxel_size = data_point['voxel_size']

            samples = get_samples(samples, voxels, voxel_size)
            np.save(out_path, samples)