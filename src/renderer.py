import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
import open3d.cuda.pybind.visualization.gui as gui

from cube import *
from tables import edge_vertices

import sympy


class RendererPLT:
    def __init__(self, interactive: bool = True):
        self.queue = []
        if interactive:
            matplotlib.use('GTK3Cairo')

        self.fig, self.ax = None, None
        self.clear()

    def clear(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.ax.w_xaxis.set_pane_color((1, 1, 1, 1))
        self.ax.w_yaxis.set_pane_color((1, 1, 1, 1))
        self.ax.w_zaxis.set_pane_color((1, 1, 1, 1))

    def render_mesh(self, vertices: [Point3], faces: [[int]], show_edges: bool = True):
        vertices = np.array(vertices)
        faces = np.array(faces)
        faces = faces.reshape(faces.shape[0], 3)
        self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles=faces, Z=vertices[:, 2], edgecolors=(0.8, 0.8, 0.8) if show_edges else None, antialiased=False, color=(1.0, 0.8, 0.6), lightsource=matplotlib.colors.LightSource(azdeg=90, altdeg=45))

    def render_surface(self, x, y, z):
        self.ax.plot_surface(x, y, z, vmin=np.min(z) * 2)

    def render_line(self, p0: Point3, p1: Point3, color=None):
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color)

    def render_cube(self, cube: Cube):
        for i in range(12):
            p0 = cube.local_to_global(cube.corners[edge_vertices[i][0]])
            p1 = cube.local_to_global(cube.corners[edge_vertices[i][1]])
            self.render_line(p0, p1, color='black')

    def show(self):
        plt.show()
        self.clear()


class RendererO3D:
    def __init__(self, resolution=(1920, 1080), legacy=True, zoom=1.0, transparent_bg=False):
        self.legacy = legacy
        self.counter = 0
        self.resolution = resolution
        self.zoom = zoom
        self.view_center = np.array([0.5, 0.5, 0.5])
        if self.legacy:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(width=self.resolution[0], height=self.resolution[1])
        else:
            gui.Application.instance.initialize()
            self.vis = o3d.visualization.O3DVisualizer(width=self.resolution[0], height=self.resolution[1])
            self.resolution = self.vis.size.width, self.vis.size.height
            # self.vis.setup_camera(field_of_view=1, center=[0.5, 0.5, 0.5], eye=[0.5, 0.5, 10], up=[0, 1, 0])
            # ext = self.vis.scene.view.get_camera().get_model_matrix()
            ext = np.eye(4)
            # ext[:, 3] = [0, 0, 20, 1]
            self.vis.setup_camera(o3d.camera.PinholeCameraIntrinsic(self.resolution[0], self.resolution[1], 500000 * zoom, 500000 * zoom, self.resolution[0] // 2, self.resolution[1] // 2), ext)
            self.vis.setup_camera(field_of_view=1, center=self.view_center, eye=self.view_center + np.array([0, 0, 500]), up=[0, 1, 0])
            # ext[0, 0] = ext[1, 1] = ext[2, 2] = 1
            self.camera = self.vis.scene.view.get_camera()
            # self.camera.set_projection(field_of_view=90, aspect_ratio=16/9, near_plane=0.01, far_plane=1000, field_of_view_type=o3d.visualization.rendering.Camera.Projection)
            # self.camera.set_projection(projection_type=o3d.visualization.rendering.Camera.Projection.Perspective, left=-1.0, right=2.0, bottom=-3.0, top=4.0, near=0.1, far=5.0)
            # self.vis.scene.add_camera(self.camera)
            # self.vis.scene.set_active_camera(self.camera)
            # self.vis.line_width = 1
            self.vis.show_skybox(False)
            self.vis.enable_raw_mode(False)
            self.vis.set_ibl("hall")
            self.vis.set_ibl_intensity(1000)
            self.vis.scene.scene.enable_sun_light(True)
            # self.vis.scene.scene.set_sun_light(direction=(0, -1, 0), color=(1, 1, 1), intensity=60000)
            self.vis.scene.scene.set_sun_light(direction=(0, -1, 0), color=(1, 1, 1), intensity=134000)#(1, 0.75, 0.47)
            # self.vis.scene.view.set_ambient_occlusion(enabled=False, ssct_enabled=False)
            self.vis.scene.view.set_shadowing(False)
            if transparent_bg:
                self.vis.set_background(np.ones(4)*100000, None)
                self.vis.scene.set_background(np.ones(4)*100000, None)
                # self.vis.scene.view.set_post_processing(False)
                # self.vis.scene.view.set_color_grading(o3d.visualization.rendering.ColorGrading(o3d.visualization.rendering.ColorGrading.Quality.MEDIUM, o3d.visualization.rendering.ColorGrading.ToneMapping.REINHARD))


        self.geometries = {}

    def __del__(self):
        if self.legacy:
            self.vis.destroy_window()
        else:
            gui.Application.instance.quit()

    def clear(self):
        for name, geometry in self.geometries.items():
            if self.legacy:
                self.vis.remove_geometry(geometry, False)
            else:
                self.vis.remove_geometry(name)
        self.geometries = {}
        self.counter = 0

    def render_mesh(self, vertices: [Point3], faces: [[int]], show_edges: bool = True, duplicate_vertices=False):
        if len(vertices) == 0 or len(faces) == 0:
            return

        if duplicate_vertices:
            new_vertices = []
            new_faces = []
            for v0, v1, v2 in faces:
                idx = len(new_vertices)
                new_faces.append((idx + 0, idx + 1, idx + 2))
                new_vertices.append(vertices[v0])
                new_vertices.append(vertices[v1])
                new_vertices.append(vertices[v2])

            vertices = np.array(new_vertices)
            faces = np.array(new_faces)

        vertices_t = self.transform(vertices)
        faces = np.array(faces)
        faces = faces.reshape(faces.shape[0], 3)

        vertices_t = Vector3dVector(vertices_t)
        triangles = Vector3iVector(faces)

        mesh = o3d.geometry.TriangleMesh(vertices_t, triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((230/230, 145/230, 70/230))#(1.0, 0.8, 0.6)
        name = f"mesh{self.counter}"
        self.counter += 1
        self.geometries[name] = mesh
        if self.legacy:
            self.vis.add_geometry(mesh, False)
        else:
            self.vis.add_geometry({"name": name, "geometry": mesh})

        if show_edges:
            wireframe = []
            for face in faces:
                last_p = vertices[face[-1]]
                for v in face:
                    wireframe.append(last_p)
                    last_p = vertices[v]
                    wireframe.append(last_p)

            self.render_lines("mesh_wireframe", wireframe, color="gray", width=0.5)

    def render_surface(self, x, y, z):
        pass

    def render_lines(self, name, points, color=None, width=1.0, cylinders=True):
        indices = np.arange(len(points)).reshape((len(points) // 2, 2))
        points = self.transform(points)

        if self.legacy and cylinders:
            for i in range(len(points) // 2):
                p0 = points[i * 2 + 0]
                p1 = points[i * 2 + 1]
                height = np.linalg.norm(p1 - p0)

                if height == 0:
                    continue

                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002 * width, height=height)
                geo_name = f"{name}_{i}_{self.counter}"
                self.counter += 1
                self.geometries[geo_name] = cyl

                up = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], dtype=np.float64)
                up /= np.linalg.norm(up)
                right = np.array([-up[1], up[0], 0])
                if np.linalg.norm(right) == 0:
                    right = np.array([-up[2], 0, up[0]])
                right /= np.linalg.norm(right)
                front = np.cross(up, right)
                R = np.eye(3)
                R[:, 2] = up
                R[:, 0] = right
                R[:, 1] = front
                cyl.rotate(R)
                cyl.translate([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2])

                cyl.compute_vertex_normals()
                if color is not None:
                    cyl.paint_uniform_color(matplotlib.colors.to_rgb(color))
                self.vis.add_geometry(cyl, False)
            return

        if cylinders:
            for i in range(len(points) // 2):
                p0 = points[i * 2 + 0]
                p1 = points[i * 2 + 1]
                height = np.linalg.norm(p1 - p0)

                if height == 0:
                    continue

                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002 * width, height=height)
                geo_name = f"{name}_{i}_{self.counter}"
                self.counter += 1
                self.geometries[geo_name] = cyl

                up = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], dtype=np.float64)
                up /= np.linalg.norm(up)
                right = np.array([-up[1], up[0], 0])
                if np.linalg.norm(right) == 0:
                    right = np.array([-up[2], 0, up[0]])
                right /= np.linalg.norm(right)
                front = np.cross(up, right)
                R = np.eye(3)
                R[:, 2] = up
                R[:, 0] = right
                R[:, 1] = front
                cyl.rotate(R)
                cyl.translate([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2])

                cyl.compute_vertex_normals()
                if color is not None:
                    cyl.paint_uniform_color(matplotlib.colors.to_rgb(color))
                self.vis.add_geometry({"name": geo_name, "geometry": cyl})
            return

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(indices),
        )

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.line_width = 0.02 * width
        mat.shader = "unlitLine"
        geo_name = f"{name}_{self.counter}"
        self.counter += 1
        self.geometries[geo_name] = line_set
        if color is not None:
            line_set.paint_uniform_color(matplotlib.colors.to_rgb(color))
        if self.legacy:
            self.vis.add_geometry(line_set, False)
        else:
            self.vis.add_geometry({"name": geo_name, "geometry": line_set, "material": mat})

    def render_points(self, name, points, color=None, size=1.0):
        points = self.transform(points)

        for i, p in enumerate(points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002 * size)
            geo_name = f"{name}_{i}_{self.counter}"
            self.counter += 1
            self.geometries[geo_name] = sphere

            sphere.translate(p)

            sphere.compute_vertex_normals()
            if color is not None:
                sphere.paint_uniform_color(matplotlib.colors.to_rgb(color))

            if self.legacy:
                self.vis.add_geometry(sphere, False)
            else:
                self.vis.add_geometry({"name": geo_name, "geometry": sphere})


    def render_cube(self, cube: Cube):
        points = []
        for i in range(12):
            p0 = cube.local_to_global(cube.corners[edge_vertices[i][0]])
            p1 = cube.local_to_global(cube.corners[edge_vertices[i][1]])
            points.append(p0)
            points.append(p1)

        self.render_lines("cube", points, color="black", width=2)

    def render_corners(self, cube: Cube):
        pos_points = []
        neg_points = []
        zero_points = []
        for i, v in enumerate(cube.values):
            corner = cube.local_to_global(cube.corners[i])
            if v > 0:
                pos_points.append(corner)
            elif v < 0:
                neg_points.append(corner)
            else:
                zero_points.append(corner)

        if len(pos_points) > 0:
            self.render_points("pos_corners", pos_points, color=[0.0, 0.5, 0.0], size=10)
        if len(neg_points) > 0:
            self.render_points("neg_corners", neg_points, color="red", size=10)
        if len(zero_points) > 0:
            self.render_points("zero_corners", zero_points, color="gray", size=10)

    def render_halfedge(self, name, p0, p1, color="red", width=2):
        self.render_lines(name, [p0, p1], color=color, width=width)

        p0 = self.transform([p0])[0]
        p1 = self.transform([p1])[0]

        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.04)

        up = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], dtype=np.float64)
        if np.linalg.norm(up) != 0:
            up /= np.linalg.norm(up)
        right = np.array([-up[1], up[0], 0], dtype=np.float64)
        if np.linalg.norm(right) == 0:
            right = np.array([-up[2], 0, up[0]], dtype=np.float64)
        right /= np.linalg.norm(right)
        front = np.cross(up, right)
        R = np.eye(3)
        R[:, 2] = up
        R[:, 0] = right
        R[:, 1] = front
        cone.rotate(R)
        cone.translate(p1)

        cone.compute_vertex_normals()
        if color is not None:
            cone.paint_uniform_color(matplotlib.colors.to_rgb(color))
        name = f"{name}_cone_{self.counter}"
        self.counter += 1
        self.geometries[name] = cone
        if self.legacy:
            self.vis.add_geometry(cone, False)
        else:
            self.vis.add_geometry({"name": name, "geometry": cone})

    def render_blue_lines(self, cube: Cube):
        iso = 0
        values = cube.values
        ui = [-1, -1]
        vi = [-1, -1]
        wi = [-1, -1]

        # a = (values[0] - values[1]) * (-values[6] + values[7] + values[4] - values[5]) - (values[4] - values[5]) * (
        #         -values[2] + values[3] + values[0] - values[1])
        # b = (iso - values[0]) * (-values[6] + values[7] + values[4] - values[5]) + (values[0] - values[1]) * (
        #         values[6] - values[4]) - (iso - values[4]) * (
        #             -values[2] + values[3] + values[0] - values[1]) - (values[4] - values[5]) * (
        #             values[2] - values[0])
        # c = (iso - values[0]) * (values[6] - values[4]) - (iso - values[4]) * (values[2] - values[0])
        #
        # d = b * b - 4 * a * c
        # if a == 0:
        #     pass  # TODO
        # if d >= 0:
        #     d = math.sqrt(d)
        #
        #     # compute u-coord of solutions
        #     ui[0] = (-b - d) / (2 * a)
        #     ui[1] = (-b + d) / (2 * a)
        #
        #     # compute v-coord of solutions
        #     g1 = values[0] * (1 - ui[0]) + values[1] * ui[0]
        #     g2 = values[2] * (1 - ui[0]) + values[3] * ui[0]
        #     vi[0] = (iso - g1) / (g2 - g1)
        #
        #     g1 = values[0] * (1 - ui[1]) + values[1] * ui[1]
        #     g2 = values[2] * (1 - ui[1]) + values[3] * ui[1]
        #     vi[1] = (iso - g1) / (g2 - g1)
        #
        #     # compute w-coordinates of solutions
        #     g1 = values[0] * (1 - ui[0]) + values[1] * ui[0]
        #     g2 = values[4] * (1 - ui[0]) + values[5] * ui[0]
        #     wi[0] = (iso - g1) / (g2 - g1)
        #
        #     g1 = values[0] * (1 - ui[1]) + values[1] * ui[1]
        #     g2 = values[4] * (1 - ui[1]) + values[5] * ui[1]
        #     wi[1] = (iso - g1) / (g2 - g1)

        # alternative:
        f0, f1, f2, f3, f4, f5, f6, f7 = values
        a = f0 - f1 - f2 + f3
        b = -f0 + f1
        c = -f0 + f2
        d = f0
        e = f0 - f1 - f2 + f3 - f4 + f5 + f6 - f7
        f = -f0 + f1 + f4 - f5
        g = -f0 + f2 + f4 - f6
        h = f0 - f4
        det = a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2
        num_u = -a * h + b * g - c * f + d * e
        num_v = -a * h - b * g + c * f + d * e
        den_u = 2 * (a * f - b * e)
        den_v = 2 * (a * g - c * e)
        if det >= 0  and den_u != 0 and den_v != 0:
            det = math.sqrt(det)
            ui[0] = (num_u - det) / den_u
            ui[1] = (num_u + det) / den_u
            vi[0] = (num_v - det) / den_v
            vi[1] = (num_v + det) / den_v
            if abs(e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h) != 0:
                wi[1] = (a * ui[0] * vi[0] + b * ui[0] + c * vi[0] + d) / (
                            e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h)
            if abs(e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h) != 0:
                wi[0] = (a * ui[1] * vi[1] + b * ui[1] + c * vi[1] + d) / (
                            e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h)

            ui[1], ui[0] = ui[0], ui[1]

        lines = [((ui[0], vi[0], 0), (ui[0], vi[0], 1)),
                 ((0, vi[0], wi[1]), (1, vi[0], wi[1])),
                 ((ui[1], 0, wi[1]), (ui[1], 1, wi[1])),
                 ((ui[1], vi[1], 0), (ui[1], vi[1], 1)),
                 ((0, vi[1], wi[0]), (1, vi[1], wi[0])),
                 ((ui[0], 0, wi[0]), (ui[0], 1, wi[0]))]

        points = []
        for p0, p1 in lines:
            if not (-0.00001 <= p0[0] <= 1.00001 and -0.00001 <= p0[1] <= 1.00001 and -0.00001 <= p0[2] <= 1.00001):
                continue
            if not (-0.00001 <= p1[0] <= 1.00001 and -0.00001 <= p1[1] <= 1.00001 and -0.00001 <= p1[2] <= 1.00001):
                continue
            points.append(cube.local_to_global(p0))
            points.append(cube.local_to_global(p1))

        if len(points) != 0:
            self.render_lines("blues", points, color=[0/255, 30/255, 255/255], width=1.9)

    def render_hexagon(self, cube: Cube):
        iso = 0
        values = cube.values
        ui = [-1, -1]
        vi = [-1, -1]
        wi = [-1, -1]

        f0, f1, f2, f3, f4, f5, f6, f7 = values
        a = f0 - f1 - f2 + f3
        b = -f0 + f1
        c = -f0 + f2
        d = f0
        e = f0 - f1 - f2 + f3 - f4 + f5 + f6 - f7
        f = -f0 + f1 + f4 - f5
        g = -f0 + f2 + f4 - f6
        h = f0 - f4
        det = a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2
        num_u = -a * h + b * g - c * f + d * e
        num_v = -a * h - b * g + c * f + d * e
        den_u = 2 * (a * f - b * e)
        den_v = 2 * (a * g - c * e)
        if det >= 0  and den_u != 0 and den_v != 0:
            det = math.sqrt(det)
            ui[0] = (num_u - det) / den_u
            ui[1] = (num_u + det) / den_u
            vi[0] = (num_v - det) / den_v
            vi[1] = (num_v + det) / den_v
            if abs(e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h) > 1e-6:
                wi[1] = (a * ui[0] * vi[0] + b * ui[0] + c * vi[0] + d) / (
                            e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h)
            if abs(e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h) > 1e-6:
                wi[0] = (a * ui[1] * vi[1] + b * ui[1] + c * vi[1] + d) / (
                            e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h)

            ui[1], ui[0] = ui[0], ui[1]

        lines = [((ui[0], vi[0], wi[0]), (ui[0], vi[0], wi[1])),
                 ((ui[0], vi[0], wi[1]), (ui[1], vi[0], wi[1])),
                 ((ui[1], vi[0], wi[1]), (ui[1], vi[1], wi[1])),
                 ((ui[1], vi[1], wi[0]), (ui[1], vi[1], wi[1])),
                 ((ui[0], vi[1], wi[0]), (ui[1], vi[1], wi[0])),
                 ((ui[0], vi[0], wi[0]), (ui[0], vi[1], wi[0]))]

        points = []
        for p0, p1 in lines:
            if not (-0.000001 <= p0[0] <= 1.000001 and -0.000001 <= p0[1] <= 1.000001 and -0.000001 <= p0[2] <= 1.000001):
                continue
            if not (-0.000001 <= p1[0] <= 1.000001 and -0.000001 <= p1[1] <= 1.000001 and -0.000001 <= p1[2] <= 1.000001):
                continue
            points.append(cube.local_to_global(p0))
            points.append(cube.local_to_global(p1))

        if len(points) != 0:
            self.render_lines("hexagon", points, color="red", width=4)#2.1

    def key_callbacks(self, callbacks):
        if self.legacy:
            for name, key, callback in callbacks:
                self.vis.register_key_action_callback(key, callback)
        else:
            for name, key, callback in callbacks:
                self.vis.add_action(name, lambda vis, f=callback: f(vis, 0, None))

    def show(self):
        if self.legacy:
            self.vis.reset_view_point(True)
            render = self.vis.get_render_option()
            render.mesh_show_back_face = True
            render.line_width = 10
            view = self.vis.get_view_control()
            view.change_field_of_view(-90)
            view.set_zoom(0.9 * self.zoom)
            view.set_front([0.5, 0.5, 500.5])
            self.vis.update_renderer()
        else:
            gui.Application.instance.add_window(self.vis)

    def run(self, blocking=True):
        tick = True
        if self.legacy:
            while self.vis.poll_events() and (blocking or tick):
                view = self.vis.get_view_control()
                view.set_up(np.array([0, 1, 0]))
                self.vis.update_renderer()
                tick = False
        else:
            while gui.Application.instance.run_one_tick() and (blocking or tick):
                model = self.camera.get_model_matrix()
                pos = model[:3, 3] - np.array([0.5, 0.5, 0.5])
                # r = np.linalg.norm(pos)
                # theta = np.arccos(pos[2] / r)
                # phi = np.arctan2(pos[1], pos[0])
                # r = 500
                # pos = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
                pos += np.array([0.5, 0.5, 0.5])

                # self.camera.look_at(center=self.view_center, eye=pos, up=[0, 1, 0])
                self.vis.scene.scene.set_sun_light(direction=-pos, color=(1, 1, 1), intensity=134000)#(1, 0.75, 0.47)
                tick = False

    def update(self):
        for name, geometry in self.geometries.items():
            if self.legacy:
                self.vis.update_geometry(geometry)
            else:
                self.vis.post_redraw()

    def save_to_file(self, filename):
        if self.legacy:
            self.vis.capture_screen_image(filename, do_render=False)
            # image = self.vis.capture_screen_float_buffer(False)
            # plt.imsave(filename, np.asarray(image), dpi=1)
        else:
            image = gui.Application.instance.render_to_image(self.vis.scene, self.vis.size.width, self.vis.size.height)
            o3d.io.write_image(filename, image)
            # self.vis.export_current_image(filename)

    def get_camera(self):
        if self.legacy:
            view = self.vis.get_view_control()
            pinhole = view.convert_to_pinhole_camera_parameters()
            return pinhole.extrinsic[:3, 3]
        else:
            return self.camera.get_model_matrix()[:3, 3]

    def set_camera(self, pos, center=None):
        if center is not None:
            self.view_center = center
        if self.legacy:
            view = self.vis.get_view_control()
            view.set_front(pos)
        else:
            self.camera.look_at(center=self.view_center, eye=pos, up=[0, 1, 0])

    def transform(self, vertices: [Point3]):
        vertices = np.array(vertices)
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] = 1 - vertices[:, 2]
        return vertices
