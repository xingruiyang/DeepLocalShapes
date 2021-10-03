import json
import os
import random

import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

instance_id = {
    'sofa': '04256520',
    # 'airliner': '02691156',
    # 'lamp': '03636649',
    'chair': '03001627',
    # 'table': '04379243'
}

num_samples_each = 6


def to_np_array(o3d_pcd):
    return np.asarray(o3d_pcd.points)


def merge_pcd(pcd1, pcd2):
    return to_o3d(np.concatenate([to_np_array(pcd1), to_np_array(pcd2)], axis=0))


def to_o3d(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def sample_sdf(mesh, num_points, voxel_size=0.05):
    num_surface_samples = num_points // 50 * 20
    num_random_samples = num_points - num_surface_samples * 2
    print(num_points, num_surface_samples*2, num_random_samples)
    surface_pts = mesh.sample_points_uniformly(num_surface_samples)

    voxels = to_np_array(surface_pts) // voxel_size
    voxels = np.unique(voxels, axis=0)
    # voxels += 0.5
    voxels *= voxel_size

    geometries = [mesh]
    for i in range(voxels.shape[0]):
        voxel = voxels[i, ...]
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.zeros((3, )) * voxel_size + voxel,
            np.ones((3, )) * voxel_size + voxel
        )
        geometries.append(bbox)

    o3d.visualization.draw_geometries(geometries)
    # scene = o3d.t.geometry.RaycastingScene()
    # scene.add_triangles(mesh.vertices, mesh.triangles)  # we do not need the geometry ID for mesh
    # signed_distance = scene.compute_signed_distance(query_points)
    # display_sdf(query_points.numpy(), signed_distance,numpy())


if __name__ == '__main__':
    base_dir = "dataset/ShapeNetCore.v2"
    out_path = "dataset/3DWareHouse"
    os.makedirs(out_path, exist_ok=True)
    id = 0
    for key, value in instance_id.items():
        instance_dir = os.path.join(base_dir, value)
        instance_out = os.path.join(out_path, key)
        os.makedirs(instance_out, exist_ok=True)
        instance_models = [f for f in os.listdir(
            instance_dir) if os.path.isdir(os.path.join(instance_dir, f))]
        num_models = len(instance_models)
        # print("{}: {}".format(key, num_models))
        # rand_choices = random.sample(instance_models, num_samples_each)
        num_each=20
        for ind in range(30,len(instance_models)):
            model = instance_models[ind]
            try:
                model_path = os.path.join(
                    instance_dir, model, 'models/model_normalized.obj')
                print(model_path)
                mesh = o3d.io.read_triangle_mesh(model_path)
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh])
                # ans = input("[y]es or [n]o:")
                # if ans == 'y':
                #     # transform = np.eye(4)
                #     # transform[:3, :3] = R.random().as_matrix()
                #     # mesh.transform(transform)
                #     # points, sdf = sample_sdf(mesh, 1000000)
                #     o3d.io.write_triangle_mesh(os.path.join(
                #         instance_out, "{}.ply".format(id)), mesh, print_progress=True)
                #     id += 1
                #     if id >= num_each:
                #         break
            except Exception as e:
                print("error processing {} for reason {}".format(model_path, str(e)))
