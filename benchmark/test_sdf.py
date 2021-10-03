import trimesh
import open3d as o3d
import argparse
import numpy as np


def to_np_array(o3d_pcd):
    return np.asarray(o3d_pcd.points)


def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def display_sdf(pts, sdf):
    color = np.zeros_like(pts)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    trimesh.PointCloud(pts, color).show()


def test_sdf(mesh, num_points, voxel_size=0.05):
    surface_pts = mesh.sample_points_uniformly(num_points)
    surface_pts = to_np_array(surface_pts)

    voxels = surface_pts // voxel_size
    voxels = np.unique(voxels, axis=0)
    voxels += .5
    voxels *= voxel_size

    for i in range(voxels.shape[0]):
        voxel = voxels[i, ...]
        bbox1 = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size,
            np.ones((3, )) * .5 * voxel_size
        )
        bbox2 = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * 1.5 * voxel_size,
            np.ones((3, )) * 1.5 * voxel_size
        )
        points = surface_pts - voxel
        dist = np.linalg.norm(points, ord=np.inf, axis=-1) 
        points = points[dist < 1.5 * voxel_size, :]
        points = to_o3d(points)
        o3d.visualization.draw_geometries([bbox1, bbox2, points])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--num-points', type=int, default=2**14)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    test_sdf(mesh, args.num_points, args.voxel_size)
    # o3d.visualization.draw_geometries([mesh])
