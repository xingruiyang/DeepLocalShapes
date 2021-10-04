import pickle
import os
import trimesh
import argparse

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def to_np_array(o3d_pcd, return_normal=False):
    if return_normal:
        return np.asarray(o3d_pcd.points), np.asarray(o3d_pcd.normals)
    else:
        return np.asarray(o3d_pcd.points)


def to_o3d(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_sdf(pcd, normal, points):
    sample_count = 11
    kd_tree = KDTree(pcd)
    sdf, indices = kd_tree.query(points, k=sample_count)
    closest_points = pcd[indices]
    dir_from_surface = points[:, None, :] - closest_points
    inside = np.einsum('ijk,ijk->ij', dir_from_surface, normal[indices]) < 0
    inside = np.sum(inside, axis=1) > (sample_count * 0.6)
    sdf = sdf[:, 0]
    sdf[inside] *= -1
    return sdf


def display_sdf(points, sdf):
    color = np.zeros_like(points)
    color[sdf > 0, 0] = 1
    color[sdf < 0, 2] = 1
    trimesh.PointCloud(points, color).show()


def get_samples(pcd, voxel_size):
    pcd, normal = to_np_array(pcd, return_normal=True)

    query_points = []
    query_points.append(pcd+np.random.randn(*pcd.shape)*0.0025)
    query_points.append(pcd+np.random.randn(*pcd.shape)*0.00025)
    query_points.append((np.random.rand(1024, 3)*3-1.5)*voxel_size)
    query_points = np.concatenate(query_points, axis=0)

    sdf = get_sdf(pcd, normal, query_points)
    # display_sdf(query_points, sdf)
    return query_points, sdf


def sample_voxels(mesh, num_points, voxel_size=0.05):
    surface_pts = mesh.sample_points_uniformly(num_points)

    voxels = to_np_array(surface_pts) // voxel_size
    voxels = np.unique(voxels, axis=0)
    voxels += .5
    voxels *= voxel_size

    num_voxels = voxels.shape[0]
    print("number voxels: {}".format(num_voxels))
    num_per_row = 1
    for i in range(1, num_voxels):
        if i * i >= num_voxels:
            num_per_row = i
            break

    geometries = []
    samples = []
    rotations = []
    centroids = []
    for i in range(voxels.shape[0]):
        print(i)
        voxel = voxels[i, ...]
        bbox_inner = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * .5 * voxel_size + voxel,
            np.ones((3, )) * .5 * voxel_size + voxel
        )
        bbox_outter = o3d.geometry.AxisAlignedBoundingBox(
            -np.ones((3, )) * 1.5 * voxel_size + voxel,
            np.ones((3, )) * 1.5 * voxel_size + voxel
        )

        y = i // num_per_row
        x = i - y * num_per_row
        # transform = np.eye(4)
        # transform[0, 3] = x
        # transform[1, 3] = y

        sub_mesh = mesh.crop(bbox_outter)
        pcd = sub_mesh.sample_points_uniformly(8192)
        pcd.translate(-voxel)
        bbox_inner.translate(-voxel)
        bbox_outter.translate(-voxel)

        # pcd.scale(2)
        # bbox_inner.scale(2)
        # bbox_outter.scale(2)

        # pcd.translate(np.array([y, x, 0]))


        points, sdf = get_samples(pcd, voxel_size)
        colors = np.zeros_like(points)
        colors[sdf > 0, 0] = 1
        colors[sdf < 0, 2] = 1
        pcd = to_o3d(points, colors)
        # pcd.scale(2)
        # bbox_inner.scale(2)
        # bbox_outter.scale(2)

        sample = np.zeros((points.shape[0], 6))
        sample[:, 0] = i
        sample[:, 1:4] = points
        sample[:, 4] = sdf
        sample[:, 5] = 1
        samples.append(sample.astype(np.float32))
        centroids.append(np.zeros(3,))
        rotations.append(np.eye(3))

        pcd.translate(np.array([y, x, 0]))
        bbox_inner.translate(np.array([y, x, 0]))
        bbox_outter.translate(np.array([y, x, 0]))

        geometries += [pcd, bbox_inner, bbox_outter]

        # o3d.visualization.draw_geometries([pcd, bbox_inner, bbox_outter])
    o3d.visualization.draw_geometries(geometries)
    return  samples, voxels, \
        np.stack(centroids, axis=0), np.stack(rotations, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    args = parser.parse_args()
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    samples, voxels, centroids, rotations = sample_voxels(mesh, 500000)

    os.makedirs(args.output, exist_ok=True)
    out = dict()
    sample_name = 'samples.npy'
    out['samples'] = sample_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name), samples)
