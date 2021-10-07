import argparse
import os
import open3d as o3d
import numpy as np


def text_3d(text, pos=None, direction=None, degree=0.0,
            font='DejaVuSansMono.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(
        img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def main(path):
    dirs = os.listdir(path)
    num_models = len(dirs)

    for i in range(max(1,num_models//100)):
        start = i * 100
        end = min((i+1) * 100, num_models)
        num_subset = end - start + 1
        num_per_row = 1
        for i in range(num_subset):
            if i * i >= num_subset:
                num_per_row = i
                break

        meshes = []
        for i in range(start, end):
            model_path = os.path.join(path, dirs[i], 'aligned/ckpt_99_mesh.ply')
            print(i, dirs[i])
            mesh = o3d.io.read_triangle_mesh(model_path)
            mesh.compute_vertex_normals()
            y = i // num_per_row
            x = i - y * num_per_row
            transform = np.eye(4)
            transform[0, 3] = y
            transform[1, 3] = x
            mesh.transform(transform)
            meshes.append(mesh)
            meshes.append(text_3d(str(i), np.array(
                [y-0.5, x+0.5, 0]), degree=-90))

        o3d.visualization.draw_geometries(meshes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    path = args.path
    main(path)
