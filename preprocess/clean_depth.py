
import argparse
import glob
import os
from itertools import chain

import cv2
import natsort
import numpy as np
import py_vmapping
import pyrender
import trimesh


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


def display_map(map):
    verts, norms = map.get_polygon()
    verts = verts.reshape(verts.shape[0]//3, 3)
    norms = norms.reshape(norms.shape[0]//3, 3)
    faces = np.arange(verts.shape[0])
    faces = faces.reshape(faces.shape[0]//3, 3)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=norms)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
    scene.add(light, np.eye(4))
    pyrender.Viewer(scene, viewer_flags={'use_direct_lighting': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--sample_method', type=str, default='uniform')
    parser.add_argument('--show_mesh', action='store_true')
    parser.add_argument('--est_traj', type=str, default='groundtruth.txt')
    args = parser.parse_args()

    scene_list = [
        # '7-scenes-redkitchen',
        # 'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1'
    ]

    num_frame_frag = 50
    for scene_name in scene_list:
        print("processing {}".format(scene_name))
        scene_path = os.path.join(args.dataset, scene_name)
        intr_filename = os.path.join(scene_path, 'camera-intrinsics.txt')
        intr = np.loadtxt(intr_filename)
        seq_list = [f for f in os.listdir(
                    scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        seq_list = natsort.natsorted(seq_list)

        map = py_vmapping.map(640, 480, intr)
        map.set_depth_scale(1000.0)
        map.create_map(500000, 450000, args.voxel_size)
        frag_id = 0
        for seq_id in range(min(len(seq_list), 3)):
            print("processing seq {}".format(seq_list[seq_id]))
            seq_name = seq_list[seq_id]
            depth_imgs = glob.glob(os.path.join(
                scene_path, seq_name, "*.depth.png"))
            depth_imgs = natsort.natsorted(depth_imgs)
            frame_selector = get_frame_selector(
                scene_name, len(depth_imgs), num_frame_frag)
            for frame_begin in frame_selector:
                print('processing fragment {}'.format(frag_id))
                map.reset()
                for i in range(num_frame_frag):
                    frame_id = frame_begin + i
                    depth = depth_imgs[frame_id]
                    pose_filename = os.path.join(
                        scene_path, seq_name, 'frame-{:06d}.pose.txt'.format(frame_id))
                    pose = np.loadtxt(pose_filename)
                    depth = cv2.imread(depth, -1)
                    map.fuse_depth(depth, pose)
                for i in range(num_frame_frag):
                    frame_id = frame_begin + i
                    pose_filename = os.path.join(
                        scene_path, seq_name, 'frame-{:06d}.pose.txt'.format(frame_id))
                    pose = np.loadtxt(pose_filename)
                    vmap = map.get_depth(pose)
                    cv2.imshow("vmap", vmap)
                    cv2.waitKey(1)
                    traced_depth = vmap[..., -2]
                    cv2.imwrite(
                        os.path.join(scene_path, seq_name,
                                     'frame-{:06d}.depth.png'.format(frame_id)),
                        (traced_depth*1000).astype(np.uint16))
                # display_map(map)
                frag_id += 1
