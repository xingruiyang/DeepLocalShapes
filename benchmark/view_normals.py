import open3d as o3d
import numpy as np
import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('mesh')
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])