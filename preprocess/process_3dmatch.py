import pickle
import glob
import numpy as np
import natsort
import os
from depth_sampler import DepthSampler
from make_voxels import Voxelizer


import argparse
from itertools import chain


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--frames-per-frag', type=int, default=50)
    parser.add_argument('--skip-frames', type=int, default=5)
    parser.add_argument('--depth-limit', type=float, default=10)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--mnfld-pnts', type=int, default=4096)
    parser.add_argument('--network', type=str, default=None)
    args = parser.parse_args()

    scene_list = [
        '7-scenes-redkitchen']
    # 'sun3d-hotel_umd-maryland_hotel3',
    # 'sun3d-mit_76_studyroom-76-1studyroom2',
    # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
    # 'sun3d-home_at-home_at_scan1_2013_jan_1',
    # 'sun3d-home_md-home_md_scan9_2012_sep_30',
    # 'sun3d-hotel_uc-scan3',
    # 'sun3d-hotel_umd-maryland_hotel1']

    for scene_name in scene_list:
        scene_path = os.path.join(args.path, scene_name)
        print("processing {}".format(scene_path))
        seq_list = [f for f in os.listdir(
            scene_path) if os.path.isdir(os.path.join(scene_path, f)) and f != 'alt']
        seq_list = natsort.natsorted(seq_list)
        intr_path = os.path.join(scene_path, 'camera-intrinsics.txt')
        intr = np.loadtxt(intr_path)

        frag_idx = 0
        for seq_id in range(min(len(seq_list), 3)):
            print("processing seq {}".format(seq_list[seq_id]))
            seq_name = seq_list[seq_id]
            depth_imgs = glob.glob(os.path.join(
                scene_path, seq_name, "*.depth.png"))
            depth_imgs = natsort.natsorted(depth_imgs)
            frame_selector = get_frame_selector(
                scene_name, len(depth_imgs), args.frames_per_frag)
            for ind in frame_selector:
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
                voxelizer = Voxelizer(
                    point_cloud, args.network,
                    args.mnfld_pnts)

                seq_out = os.path.join(
                    args.output, scene_name, str(frag_idx))
                os.makedirs(seq_out, exist_ok=True)

                samples = voxelizer.create_voxels(
                    args.voxel_size, out_path=seq_out)
                samples, voxels, centroids, rotations, surface = samples

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

                os.makedirs(seq_out, exist_ok=True)
                with open(os.path.join(seq_out, "samples.pkl"), "wb") as f:
                    pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
                np.save(os.path.join(seq_out, sample_name),
                        samples.astype(np.float32))
                np.save(os.path.join(seq_out, surface_pts_name),
                        surface.astype(np.float32))

                frag_idx += 1
