import argparse
import random
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


class SceneGenerator():
    def __init__(self, num_shapes=10):
        self.radius = 1
        self.num_shapes = num_shapes
        self.shape_type = [
            'cuboid',  'cylinder', 'ellipsoid']
        # self.shape_type = ['cuboid']

    def random_pose(self):
        rotation = Rotation.random().as_matrix()
        pose = np.eye(4)
        pose[:3, 3] = [
            random.uniform(-self.radius, self.radius),
            random.uniform(-self.radius, self.radius),
            random.uniform(-self.radius, self.radius)
        ]
        pose[:3, :3] = rotation
        return pose

    def random_cuboid(self):
        cuboid = trimesh.creation.box(
            extents=[
                random.uniform(self.radius*0.1, self.radius*0.3),
                random.uniform(self.radius*0.1, self.radius*0.3),
                random.uniform(self.radius*0.1, self.radius*0.3),
            ])
        return cuboid

    def random_cylinder(self):
        return trimesh.creation.cylinder(
            radius=random.uniform(self.radius*0.1, self.radius*0.12),
            height=random.uniform(self.radius*0.2, self.radius*0.4))

    def random_cone(self):
        return trimesh.creation.cone(
            radius=random.uniform(self.radius*0.07, self.radius*0.15),
            height=random.uniform(self.radius*0.2, self.radius*0.4))

    def random_sphere(self):
        return trimesh.primitives.Sphere(
            radius=random.uniform(self.radius*0.08, self.radius*0.16))

    def random_ellipsoid(self):
        sphere = trimesh.primitives.Sphere(radius=self.radius * 0.05)
        verts = sphere.vertices
        new_verts = np.zeros_like(verts)
        new_verts[:, 0] = verts[:, 0] * random.uniform(1, 5)
        new_verts[:, 1] = verts[:, 1] * random.uniform(1, 5)
        new_verts[:, 2] = verts[:, 2] * random.uniform(1, 5)
        return trimesh.Trimesh(new_verts, sphere.faces)

    def scale_to_unit_cube(self, mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        vertices = mesh.vertices - mesh.bounding_box.centroid
        vertices *= 2 / np.max(mesh.bounding_box.extents)
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    def generate_scene(self):
        start = time.time()
        scene = trimesh.scene.scene.Scene(camera=None)
        for _ in range(self.num_shapes):
            shape_fn = random.choice(self.shape_type)
            if shape_fn == 'cuboid':
                shape = self.random_cuboid()
            if shape_fn == 'cylinder':
                shape = self.random_cylinder()
            if shape_fn == 'cone':
                shape = self.random_cone()
            if shape_fn == 'sphere':
                shape = self.random_sphere()
            if shape_fn == 'ellipsoid':
                shape = self.random_ellipsoid()
            transform = self.random_pose()
            # shape.visual.face_colors = trimesh.visual.random_color()
            scene.add_geometry(shape, transform=transform)
        print("Scene generation took {:.2f} s".format(time.time()-start))
        return self.scale_to_unit_cube(scene)
        # return scene.dump(True), self.scale_to_unit_cube(scene)


# random_ellipsoid().show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shapes', type=int, default=10)
    parser.add_argument('--export', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    mesh = SceneGenerator(num_shapes=args.num_shapes).generate_scene()

    if args.show:
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        bounding_sphere = trimesh.primitives.Box(extents=[2, 2, 2])
        points = bounding_sphere.sample(2**10)
        points = trimesh.PointCloud(points)
        scene.add_geometry(points)
        scene.show()

    if args.export is not None:
        mesh.export(args.export)
        print("Mesh is generated to {}".format(args.export))
