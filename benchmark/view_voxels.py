import open3d as o3d
import argparse
import numpy as np


def to_np_array(o3d_pcd):
    return np.asarray(o3d_pcd.points)


def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def view_voxels(mesh, num_points, voxel_size=0.05):
    surface_pts = mesh.sample_points_uniformly(num_points)

    voxels = to_np_array(surface_pts) // voxel_size
    voxels = np.unique(voxels, axis=0)
    voxels += .5
    voxels *= voxel_size

    # geometries = [surface_pts]
    for i in range(voxels.shape[0]):
        voxel = voxels[i, ...]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        bbox_outter = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * 1.5 * voxel_size + voxel,
            np.ones((3, )) * 1.5 * voxel_size + voxel
        )
        sub_mesh = surface_pts.crop(bbox_outter)
        # geometries.append(bbox)
        grid=o3d.geometry.VoxelGrid.create_from_point_cloud(sub_mesh, 3*voxel_size/256)
        o3d.visualization.draw_geometries([grid, bbox_inner, bbox_outter])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--num-points', type=int, default=2**14)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    view_voxels(mesh, args.num_points, args.voxel_size)
    # o3d.visualization.draw_geometries([mesh])
