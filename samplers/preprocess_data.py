import pickle
import argparse
import numpy as np
import os
import trimesh

def display_sdf(pts, sdf):
    color=np.zeros_like(pts)
    color[sdf>0, 0]=1
    color[sdf<0, 2]=1
    trimesh.PointCloud(pts, color).show()

def split_samples(samples, surface_pts, voxel_size=0.05):
    points = samples[:, :3]
    sdf = samples[:, 3]
    # surface_pts = model.dump(True).sample(100000)

    display_sdf(points, sdf)
    # trimesh.Scene([trimesh.PointCloud(points), trimesh.PointCloud(surface_pts)]).show()

    voxels = surface_pts // voxel_size
    voxels = np.unique(voxels, axis=0)
    voxels += .5
    voxels *= voxel_size
    print("{} voxels to be sampled".format(voxels.shape[0]))
    samples = []
    centroids = []
    rotations = []

    surface = []
    rand_surface = []
    rand_sdf = []

    for vid in range(voxels.shape[0]):
        print("{}/{}".format(vid, voxels.shape[0]))
        voxel = voxels[vid, :]
        centroid = np.zeros((3,))
        rotation = np.eye(3)

        pcd = points - voxel
        dist = np.linalg.norm(pcd, ord=np.inf, axis=-1)
        selector = dist < (1.5 * voxel_size)
        query_points = pcd[selector, :]
        query_sdf = sdf[selector]

        # display_sdf(pcd, sdf)
        print(query_points.shape)

        vsample = np.zeros((query_points.shape[0], 6))
        vsample[:, 0] = float(vid)
        vsample[:, 1:4] = query_points
        vsample[:, 4] = query_sdf
        vsample[:, 5] = 1
        samples.append(vsample)
        centroids.append(centroid)
        rotations.append(rotation)

    samples = np.concatenate(samples, axis=0)
    centroids = np.stack(centroids, axis=0)
    rotations = np.stack(rotations, axis=0)

    print("{} points sampled with {} voxels aligned".format(
        samples.shape[0], rotations.shape[0]))

    return samples, voxels, centroids, rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy', type=str)
    parser.add_argument('surface', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    args = parser.parse_args()

    voxel_size = args.voxel_size
    npz = np.load(args.npy)
    surface = np.load(args.surface)
    samples, voxels, centroids, rotations = split_samples(
        npz, surface, voxel_size)

    sample_name = 'samples.npy'
    surface_pts_name = 'surface_pts.npy'
    surface_sdf_name = 'surface_sdf.npy'

    out = dict()
    out['samples'] = sample_name
    out['surface_pts'] = surface_pts_name
    out['surface_sdf'] = surface_sdf_name
    out['voxels'] = voxels.astype(np.float32)
    out['centroids'] = centroids.astype(np.float32)
    out['rotations'] = rotations.astype(np.float32)
    out['voxel_size'] = args.voxel_size

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "samples.pkl"), "wb") as f:
        pickle.dump(out, f,  pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(args.output, sample_name),
            samples.astype(np.float32))
    np.save(os.path.join(args.output, surface_pts_name),
            surface.astype(np.float32))