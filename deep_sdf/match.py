import argparse
import json
import pickle

import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree
from tqdm import tqdm

from network import ImplicitNet
from utils import load_model


def has(arg):
    return arg is not None


def to_o3d_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def to_o3d_feat(arr):
    feat = o3d.pipelines.registration.Feature()
    feat.data = arr.transpose()
    return feat


def execute_global_registration(
        src_pts, dst_pts, src_features, dst_features, distance_threshold):

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pts,
        dst_pts,
        src_features,
        dst_features,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
    return result


def gram_schmidt(rots):
    v1 = rots[..., :3]
    v1 = v1 / torch.max(torch.sqrt(torch.sum(v1**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v2 = rots[..., 3:] - \
        torch.sum(v1 * rots[..., 3:], dim=-1, keepdim=True) * v1
    v2 = v2 / torch.max(torch.sqrt(torch.sum(v2**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v3 = v1.cross(v2)

    rots = torch.stack([v1, v2, v3], dim=2)

    return rots[0, ...]


def run_icp_refine(
    network: torch.nn.Module,
    src_voxels: np.ndarray,
    src_latents: np.ndarray,
    query_pts: np.ndarray,
    voxel_size: float,
    transform: np.ndarray,
    num_iterations: int = 10,
    min_inlier_points: int = 0,
    centroids: np.ndarray = None,
    rotations: np.ndarray = None,
    use_gpu: bool = True
):
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    network.to(device).eval()

    query_pcd = torch.from_numpy(query_pts).float().to(device)
    train_latents = torch.from_numpy(src_latents).float().to(device)

    if centroids is not None:
        centroids = torch.from_numpy(centroids).float().to(device)
    if rotations is not None:
        rotations = torch.from_numpy(rotations).float().to(device)

    init_r = transform[:3, :3]
    init_t = transform[:3, 3]

    best_r = torch.from_numpy(init_r).float().to(device)
    # column-wise flatten
    best_r = best_r[:3, :2].swapaxes(0, 1).reshape(-1, 6)
    best_t = torch.from_numpy(init_t).float().to(device)

    best_r.requires_grad_()
    best_t.requires_grad_()

    train_voxels = src_voxels/voxel_size - .5
    train_voxels = np.round(train_voxels).astype(int)
    oct_tree = KDTree(train_voxels)

    optimizer = torch.optim.SGD([best_r, best_t], lr=1e-4)
    pbar = tqdm(range(num_iterations))
    for n_iter in pbar:
        best_r_mat = gram_schmidt(best_r)
        query_pts = torch.matmul(query_pcd - best_t, best_r_mat)
        # query_pts = torch.matmul(
        #     query_pcd, best_r_mat.transpose(0, 1)) + best_t
        query_np = query_pts.detach().cpu().numpy()
        query_np = query_np // voxel_size
        dist, indices = oct_tree.query(query_np, k=1)
        dist = dist.astype(int)

        indices = indices[dist == 0]
        point_indices = np.where(dist == 0)[0]
        if point_indices.shape[0] < min_inlier_points:
            print("icp failed: no sufficient inlier points {}".format(
                point_indices.shape[0]))
            break

        point_indices = torch.from_numpy(point_indices).int().to(device)
        indices = torch.from_numpy(indices).int().to(device)
        query_np = torch.from_numpy(query_np).float().to(device)
        query_np += 0.5

        points = torch.index_select(
            query_pts/voxel_size-query_np, 0, point_indices)
        latents = torch.index_select(train_latents, 0, indices)
        if has(centroids) and has(rotations):
            centroid = torch.index_select(centroids, 0, indices)
            rotation = torch.index_select(rotations, 0, indices)
            points = torch.matmul(
                (points-centroid).unsqueeze(1),
                rotation.transpose(1, 2)).squeeze()

        inputs = torch.cat([latents, points], dim=-1)
        sdf_pred = network(inputs).squeeze()

        loss = (sdf_pred.abs()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("inliers: {}/{} loss: {:.2f}".format(
            point_indices.shape[0], query_pts.shape[0], loss.item()))

    best_r = gram_schmidt(best_r)
    best_transform = np.eye(4)
    best_transform[:3, :3] = best_r.detach().cpu().numpy()
    best_transform[:3, 3] = best_t.detach().cpu().numpy()
    return best_transform


def load_data(args):
    input_data = dict()
    input_data['src_voxels'] = pickle.load(
        open(args.src_voxels, 'rb'))['voxels']
    input_data['dst_voxels'] = pickle.load(
        open(args.dst_voxels, 'rb'))['voxels']
    input_data['src_latents'] = np.load(args.src_latents)
    input_data['dst_latents'] = np.load(args.dst_latents)
    input_data['voxel_size'] = pickle.load(
        open(args.src_voxels, 'rb'))['voxel_size']

    if args.network_cfg and args.network_ckpt and args.src_mesh:
        network_args = json.load(open(args.network_cfg, 'r'))
        network = ImplicitNet(**network_args['params'])
        load_model(args.network_ckpt, network)
        input_data['network'] = network
        input_data['query_pts'] = trimesh.load(args.src_mesh).sample(100000)

    if args.orient:
        input_data['rotations'] = pickle.load(
            open(args.src_voxels, 'rb'))['rotations']
        input_data['centroids'] = pickle.load(
            open(args.src_voxels, 'rb'))['centroids']
    return input_data


def compute_rigid_transform(
        src_voxels: np.ndarray,
        dst_voxels:  np.ndarray,
        src_latents: np.ndarray,
        dst_latents: np.ndarray,
        voxel_size: float,
        query_pts: np.ndarray = None,
        network: torch.nn.Module = None,
        num_iterations: int = -1,
        centroids: np.ndarray = None,
        rotations: np.ndarray = None
):
    src_pts = to_o3d_pcd(src_voxels)
    dst_pts = to_o3d_pcd(dst_voxels)
    src_features = to_o3d_feat(src_latents)
    dst_features = to_o3d_feat(dst_latents)
    result = execute_global_registration(
        src_pts, dst_pts, src_features, dst_features, voxel_size*3)
    transform = result.transformation

    print(transform)

    if has(network) and has(query_pts):
        transform = run_icp_refine(
            network, src_voxels, src_latents,
            query_pts, voxel_size,
            transform,
            num_iterations=num_iterations,
            centroids=centroids,
            rotations=rotations,
            use_gpu=True
        )
    print(transform)

    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_voxels', type=str)
    parser.add_argument('dst_voxels', type=str)
    parser.add_argument('src_latents', type=str)
    parser.add_argument('dst_latents', type=str)
    parser.add_argument('--num-iter', type=int, default=10)
    parser.add_argument('--src-mesh', type=str, default=None)
    parser.add_argument('--dst-mesh', type=str, default=None)
    parser.add_argument('--network-cfg', type=str, default=None)
    parser.add_argument('--network-ckpt', type=str, default=None)
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    input_data = load_data(args)
    transform = compute_rigid_transform(
        **input_data,
        num_iterations=args.num_iter)

    if has(args.src_mesh) and has(args.dst_mesh) and args.show:
        geometry = []
        src_mesh = o3d.io.read_triangle_mesh(args.src_mesh)
        src_mesh.compute_vertex_normals()
        src_mesh.transform(transform)
        src_mesh.paint_uniform_color([1, 0.5, 0])
        geometry.append(src_mesh)

        dst_mesh = o3d.io.read_triangle_mesh(args.dst_mesh)
        dst_mesh.compute_vertex_normals()
        geometry.append(dst_mesh)

        o3d.visualization.draw_geometries(geometry)
