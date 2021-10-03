import argparse
import trimesh
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh)
    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices /= np.max(mesh.bounding_box.extents)

    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    if args.output is None:
        mesh.export(mesh)
    else:
        mesh.export(args.output)
