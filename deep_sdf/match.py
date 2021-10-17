import argparse
import copy
import json
import pickle

import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree
from tqdm import tqdm

from line_mesh import LineMesh
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
        False,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    # result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
    #     src_pts, dst_pts, src_features, dst_features,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=distance_threshold))
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
    dst_voxels: np.ndarray,
    dst_latents: np.ndarray,
    query_pts: np.ndarray,
    voxel_size: float,
    transform: np.ndarray,
    num_iterations: int = 10,
    min_inlier_points: int = 1,
    centroids: np.ndarray = None,
    rotations: np.ndarray = None,
    use_gpu: bool = True
):
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    network.to(device).eval()

    query_pcd = torch.from_numpy(query_pts).float().to(device)
    train_latents = torch.from_numpy(dst_latents).float().to(device)

    if centroids is not None:
        centroids = torch.from_numpy(centroids).float().to(device)
    if rotations is not None:
        rotations = torch.from_numpy(rotations).float().to(device)
    
    transform = copy.deepcopy(transform)
    init_r = transform[:3, :3]
    init_t = transform[:3, 3]
    print("before: ", transform)

    best_r = torch.from_numpy(init_r).float().to(device)
    # column-wise flatten
    best_r = best_r[:3, :2].swapaxes(0, 1).reshape(-1, 6)
    best_t = torch.from_numpy(init_t).float().to(device)

    best_r.requires_grad_()
    best_t.requires_grad_()

    train_voxels = dst_voxels/voxel_size - .5
    train_voxels = np.round(train_voxels).astype(int)
    oct_tree = KDTree(train_voxels)

    optimizer = torch.optim.Adam([best_r, best_t], lr=1e-2)
    pbar = tqdm(range(num_iterations))
    for n_iter in pbar:
        best_r_mat = gram_schmidt(best_r)
        # query_pts = torch.matmul(query_pcd - best_t, best_r_mat)
        query_pts = torch.matmul(
            query_pcd, best_r_mat.transpose(0, 1)) + best_t
            
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
                (points-centroid/voxel_size).unsqueeze(1),
                rotation.transpose(1, 2)).squeeze()

        inputs = torch.cat([latents, points], dim=-1)
        sdf_pred = network(inputs).squeeze()

        loss = (sdf_pred**2).mean()

        optimizer.zero_grad()
        loss.backward()
        print(best_t.grad)
        optimizer.step()

        pbar.set_description("inliers: {}/{} loss: {:.2f}".format(
            point_indices.shape[0], query_pts.shape[0], loss.item()))

    best_r = gram_schmidt(best_r)
    best_transform = np.eye(4)
    best_transform[:3, :3] = best_r.detach().cpu().numpy()
    best_transform[:3, 3] = best_t.detach().cpu().numpy()
    return best_transform


class LatentMatcher(object):
    def __init__(
            self,
            src_voxels: np.ndarray,
            dst_voxels:  np.ndarray,
            src_latents: np.ndarray,
            dst_latents: np.ndarray,
            voxel_size: float,
            query_pts: np.ndarray = None,
            network: torch.nn.Module = None,
            centroids: np.ndarray = None,
            rotations: np.ndarray = None,
            distance_threshold=1) -> None:
        super().__init__()
        self.src_voxels = src_voxels
        self.dst_voxels = dst_voxels
        self.src_latents = src_latents
        self.dst_latents = dst_latents
        self.src_pts = to_o3d_pcd(src_voxels)
        self.dst_pts = to_o3d_pcd(dst_voxels)
        self.src_features = to_o3d_feat(src_latents)
        self.dst_features = to_o3d_feat(dst_latents)
        self.voxel_size = voxel_size
        self.centroids = centroids
        self.rotations = rotations
        self.network = network
        self.query_pts = query_pts
        self.distance_threshold = distance_threshold * voxel_size

    def compute_rigid_transform(self):
        result = execute_global_registration(
            self.src_pts,
            self.dst_pts,
            self.src_features,
            self.dst_features,
            self.distance_threshold)
        return result

    def refine_pose(self, transform, num_iter, use_gpu=True):
        if not has(self.network):
            print("network not given!")
            return None

        return run_icp_refine(
            self.network,
            self.dst_voxels,
            self.dst_latents,
            self.query_pts,
            self.voxel_size,
            transform,
            num_iterations=num_iter,
            min_inlier_points=5000,
            centroids=self.centroids,
            rotations=self.rotations,
            use_gpu=use_gpu
        )


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

    if args.network_cfg and args.network_ckpt and args.icp:
        network_args = json.load(open(args.network_cfg, 'r'))
        network = ImplicitNet(**network_args['params'])
        load_model(args.network_ckpt, network)
        input_data['network'] = network
        input_data['query_pts'] = trimesh.load(args.src_mesh).sample(50000)

    if args.orient:
        input_data['rotations'] = pickle.load(
            open(args.dst_voxels, 'rb'))['rotations']
        input_data['centroids'] = pickle.load(
            open(args.dst_voxels, 'rb'))['centroids']
    return input_data


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
    parser.add_argument('--icp', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    input_data = load_data(args)
    matcher = LatentMatcher(**input_data)
    coarse_result = matcher.compute_rigid_transform()
    transform = coarse_result.transformation
    print(coarse_result.correspondence_set)
    final_transform = transform

    if args.icp:
        transform = matcher.refine_pose(
            transform, args.num_iter, True)

    if has(args.src_mesh) and has(args.dst_mesh) and args.show:
        geometry = []
        src_mesh = o3d.io.read_triangle_mesh(args.src_mesh)
        src_mesh.compute_vertex_normals()
        transformed_mesh = copy.deepcopy(src_mesh)

        src_mesh.paint_uniform_color([0, 0.5, 1])
        geometry.append(src_mesh)
        # if args.icp:
        #     final_mesh = copy.deepcopy(src_mesh)
        #     final_mesh.paint_uniform_color([0.5, 1, 0])
        #     final_mesh.transform(transform)
        #     geometry.append(final_mesh)
        # print(transform)
        transformed_mesh = copy.deepcopy(src_mesh)
        transformed_mesh.transform(transform)
        transformed_mesh.paint_uniform_color([0.5, 1, 0])
        geometry.append(transformed_mesh)

        # query_pts = to_o3d_pcd(input_data['query_pts'])
        # geometry.append(query_pts)

        dst_mesh = o3d.io.read_triangle_mesh(args.dst_mesh)
        dst_mesh.compute_vertex_normals()
        geometry.append(dst_mesh)

        lines = np.asarray(coarse_result.correspondence_set)
        src_pts = input_data['src_voxels']
        dst_pts = input_data['dst_voxels']
        src_pts = np.matmul(
            src_pts, transform[:3, :3].transpose()) + transform[:3, 3]
        points = np.concatenate([src_pts, dst_pts], axis=0)
        lines[:, 1] += src_pts.shape[0]
        matches = LineMesh(points, lines, colors=[1, 0.2, 0], radius=0.005)
        matches_geoms = matches.cylinder_segments
        geometry += [*matches_geoms]

        o3d.visualization.draw_geometries(geometry)
