import numpy as np
import open3d as o3d
import argparse
from sklearn.neighbors import KDTree

def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def get_voxels(surface_pts, voxel_size=0.15):
    voxels = surface_pts // voxel_size
    voxels = np.unique(voxels, axis=0)
    voxels += .5
    voxels *= voxel_size
    return voxels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd')
    parser.add_argument('--voxel_size', type=float, default=0.1)
    args = parser.parse_args()
    num_voxels = 1000
    pcd = np.load(args.pcd)
    samples = pcd[np.random.permutation(pcd.shape[0])[:num_voxels], :]
    surface_pts = to_o3d(pcd)
    geometries = []
    voxels = get_voxels(pcd, args.voxel_size)
    kdtree = KDTree(pcd)
    # dist, ind = kdtree.query(voxels)
    # voxels=pcd[ind,:].astype(float).squeeze()
    # num_voxels=voxels.shape[0]

    for i in range(num_voxels):
        voxel = samples[i, :]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * 1.5 * args.voxel_size + voxel,
            np.ones((3, )) * 1.5 * args.voxel_size + voxel
        )
        # bbox_outter = o3d.geometry.AxisAlignedBoundingBox(
        #     -np.ones((3, )) * 1.5 * voxel_size + voxel,
        #     np.ones((3, )) * 1.5 * voxel_size + voxel
        # )
        # sub_mesh = surface_pts.crop(bbox_outter)
        geometries.append(bbox_inner)
        # grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        #     sub_mesh, 3*voxel_size/256)
    o3d.visualization.draw_geometries(geometries+[surface_pts])
