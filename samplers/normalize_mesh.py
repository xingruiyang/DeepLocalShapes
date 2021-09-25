import mesh_to_sdf
import argparse
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    if args.output is None:
        mesh.export(mesh)
    else:
        mesh.export(args.output)
