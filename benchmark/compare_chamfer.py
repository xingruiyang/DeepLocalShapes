import trimesh
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors


def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError(
            "Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input1', type=str)
    parser.add_argument('input2', type=str)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_samples', type=int, default=2**16)
    args = parser.parse_args()

    mesh1 = trimesh.load(args.input1)

    if args.normalize:
        mesh1 = scale_to_unit_cube(mesh1)
    mesh2 = trimesh.load(args.input2)

    point_cloud1 = mesh1.sample(args.num_samples)
    point_cloud2 = mesh2.sample(args.num_samples)
    print("1to2: {} 2to1: {} bi: {}".format(
        chamfer_distance(point_cloud1, point_cloud2, direction='x_to_y'),
        chamfer_distance(point_cloud1, point_cloud2, direction='y_to_x'),
        chamfer_distance(point_cloud1, point_cloud2, direction='bi')))
