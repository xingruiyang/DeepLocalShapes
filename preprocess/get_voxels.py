import open3d as o3d
import numpy as np
import argparse
from sklearn.neighbors import KDTree


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
        voxels = np.random.permutation(cloud.shape[0])[:voxel_num]
        voxels = cloud[voxels, :]
        return voxels
    elif method == 'closest':
        voxels = cloud // voxel_size
        voxels = np.unique(voxels, axis=0)
        voxels += .5
        voxels *= voxel_size

        kd_tree = KDTree(cloud)
        dist,ind = kd_tree.query(voxels)
        voxels = cloud[ind[:,0], :]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cloud', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='uniform')
    args = parser.parse_args()

    cloud = np.load(args.cloud)[:, 1:4]
    voxels = get_voxels(cloud, voxel_size=args.voxel_size, method=args.method)
    draw_voxels(cloud, voxels, voxel_size=args.voxel_size)
