import argparse
import os
import pickle

import numpy as np
import trimesh

def display_sdf(surface_sdf):
    pts = surface_sdf[:, 0:3]
    sdf = surface_sdf[:, 3]
    colors = np.zeros_like(pts)
    colors[sdf>0, 0] = 1
    colors[sdf<0, 2] = 1
    trimesh.PointCloud(pts, colors).show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sdf', action='store_true')
    args = parser.parse_args()

    path = args.path
    index = pickle.load(open(os.path.join(path, 'samples.pkl'), 'rb'))
    if args.sdf:
        surface_sdf_name = index['surface_sdf']
        surface_sdf = np.load(os.path.join(path, surface_sdf_name))
        display_sdf(surface_sdf)
    else:
        surface_pts_name = index['surface_pts']
        surface_pts = np.load(os.path.join(path, surface_pts_name))
        trimesh.PointCloud(surface_pts).show()