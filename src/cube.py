import numpy as np
from skimage import measure
import sympy

from tables import edge_vertices, face_vertices

Point2 = (float, float)
Point3 = (float, float, float)


class Cube:
    def __init__(self, bbox: (Point3, Point3) = ((0, 0, 0), (1, 1, 1))):
        self.bbox_min = np.array(bbox[0])
        self.bbox_max = np.array(bbox[1])

        self.values = [0] * 8

        self.corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

    def __getitem__(self, item: (int, int, int)) -> float:
        return self.values[item[2] * 4 + item[1] * 2 + item[0] * 1]

    def __setitem__(self, item: (int, int, int), value: float):
        self.values[item[2] * 4 + item[1] * 2 + item[0] * 1] = value

    def trilinear(self, point: Point3) -> float:
        u, v, w = point
        return ((1 - w) * ((1 - v) * ((1 - u) * self[0, 0, 0] + u * self[1, 0, 0]) +
                                 v * ((1 - u) * self[0, 1, 0] + u * self[1, 1, 0])) +
                      w * ((1 - v) * ((1 - u) * self[0, 0, 1] + u * self[1, 0, 1]) +
                                 v * ((1 - u) * self[0, 1, 1] + u * self[1, 1, 1])))

    def trilinear_gradient(self, point: Point3):
        u, v, w = point
        # du0 = ((1 - w) * ((1 - v) * (-self[0, 0, 0] + self[1, 0, 0]) +
        #                        v * (-self[0, 1, 0] + self[1, 1, 0])) +
        #             w * ((1 - v) * (-self[0, 0, 1] + self[1, 0, 1]) +
        #                        v * (-self[0, 1, 1] + self[1, 1, 1])))
        #
        # dv0 =  ((1 - w) * (-((1 - u) * self[0, 0, 0] + u * self[1, 0, 0]) +
        #                    ((1 - u) * self[0, 1, 0] + u * self[1, 1, 0])) +
        #              w * (-((1 - u) * self[0, 0, 1] + u * self[1, 0, 1]) +
        #                    ((1 - u) * self[0, 1, 1] + u * self[1, 1, 1])))
        #
        # dw0 =  (-((1 - v) * ((1 - u) * self[0, 0, 0] + u * self[1, 0, 0]) +
        #                v * ((1 - u) * self[0, 1, 0] + u * self[1, 1, 0])) +
        #         ((1 - v) * ((1 - u) * self[0, 0, 1] + u * self[1, 0, 1]) +
        #                v * ((1 - u) * self[0, 1, 1] + u * self[1, 1, 1])))

        du = -(1 - w) * (1 - v) * self[0, 0, 0] + (1 - w) * (1 - v) * self[1, 0, 0] \
             -(1 - w) *       v * self[0, 1, 0] + (1 - w) *       v * self[1, 1, 0] \
             -      w * (1 - v) * self[0, 0, 1] +       w * (1 - v) * self[1, 0, 1] \
             -      w *       v * self[0, 1, 1] +       w *       v * self[1, 1, 1]

        dv = -(1 - w) * (1 - u) * self[0, 0, 0] - (1 - w) * u * self[1, 0, 0] \
             +(1 - w) * (1 - u) * self[0, 1, 0] + (1 - w) * u * self[1, 1, 0] \
             -      w * (1 - u) * self[0, 0, 1] -       w * u * self[1, 0, 1] \
             +      w * (1 - u) * self[0, 1, 1] +       w * u * self[1, 1, 1]

        dw = -(1 - v) * (1 - u) * self[0, 0, 0] - (1 - v) * u * self[1, 0, 0] \
             -      v * (1 - u) * self[0, 1, 0] -       v * u * self[1, 1, 0] \
             +(1 - v) * (1 - u) * self[0, 0, 1] + (1 - v) * u * self[1, 0, 1] \
             +      v * (1 - u) * self[0, 1, 1] +       v * u * self[1, 1, 1]

        return du, dv, dw

    def parametric_level_set(self, point: Point3, dimension: int, iso: float = 0) -> float:
        interpolation = self.trilinear(point)
        surface = sympy.solve(interpolation - iso, point[dimension], rational=True)[0]
        solution = surface#.factor(point[(dimension + 1) % 3], point[(dimension + 2) % 3])
        return solution

    def bilinear_face(self, face_point: Point2, face: int) -> float:
        variable = sympy.symbols('bilinear_face_x')
        value = face % 2

        if face == 0:
            point = (face_point[1], face_point[0], variable)
        elif face == 1:
            point = (1 - face_point[1], face_point[0], variable)
        elif face == 2:
            point = (face_point[1], variable, 1 - face_point[0])
        elif face == 3:
            point = (face_point[1], variable, face_point[0])
        elif face == 4:
            point = (variable, face_point[0], 1 - face_point[1])
        elif face == 5:
            point = (variable, face_point[0], face_point[1])
        else:
            raise ValueError('invalid face')

        interpolation = self.trilinear(point)
        return interpolation.subs(variable, value, doit=False)

    def hyperbola(self, face_u: float, face: int, iso: float = 0) -> float:
        hyperbola_v = sympy.symbols('hyperbola_v')
        bilinear = self.bilinear_face((face_u, hyperbola_v), face)
        return sympy.solve(bilinear - iso, hyperbola_v, rational=True)[0]

    def asymptotes(self, face: int, iso: float = 0):
        asymptotes_u, asymptotes_v = sympy.symbols('asymptotes_u asymptotes_v')
        hyperbola_u = self.hyperbola(asymptotes_u, face, iso)
        hyperbola_v = sympy.solve(hyperbola_u - asymptotes_v, asymptotes_u, rational=True)[0]
        return sympy.solve(1 / hyperbola_u, asymptotes_u, rational=True)[0], sympy.solve(1 / hyperbola_v, asymptotes_v, rational=True)[0]

    def asymptotic_decider(self, face: int, iso: float = 0):
        asymptotes_u, asymptotes_v = self.asymptotes(face, iso)
        return self.bilinear_face((asymptotes_u, asymptotes_v), face)

    def hyperbola_intersections(self, dimension: int, iso: float = 0):
        front_face = (2 - dimension) * 2
        back_face = front_face + 1
        hyperbola_intersections_u = sympy.symbols('hyperbola_intersections_u')
        hyperbola_front = self.hyperbola(hyperbola_intersections_u, front_face, iso)
        hyperbola_back = self.hyperbola(hyperbola_intersections_u, back_face, iso)
        return sympy.solve(hyperbola_front - hyperbola_back, hyperbola_intersections_u, rational=True)

    def local_to_global(self, point: Point3) -> Point3:
        return (self.bbox_max - self.bbox_min) * point + self.bbox_min

    def global_to_local(self, point: Point3) -> Point3:
        return (point - self.bbox_min) / (self.bbox_max - self.bbox_min)

    def face_to_local(self, point: Point2, face: int) -> Point3:
        v0 = face_vertices[face][0]
        v1 = face_vertices[face][1]
        v2 = face_vertices[face][2]
        v3 = face_vertices[face][3]
        p0 = self.corners[v0]
        p1 = self.corners[v1]
        p2 = self.corners[v2]
        p3 = self.corners[v3]
        u, v = point

        x = ((1 - v) * ((1 - u) * p0[0] + u * p1[0]) + v * ((1 - u) * p2[0] + u * p3[0]))
        y = ((1 - v) * ((1 - u) * p0[1] + u * p1[1]) + v * ((1 - u) * p2[1] + u * p3[1]))
        z = ((1 - v) * ((1 - u) * p0[2] + u * p1[2]) + v * ((1 - u) * p2[2] + u * p3[2]))
        return x, y, z

    def edge_to_local(self, point: float, edge: int) -> Point3:
        v0 = edge_vertices[edge][0]
        v1 = edge_vertices[edge][1]
        p0 = self.corners[v0]
        p1 = self.corners[v1]

        x = (1 - point) * p0[0] + point * p1[0]
        y = (1 - point) * p0[1] + point * p1[1]
        z = (1 - point) * p0[2] + point * p1[2]
        return x, y, z

    def trilinear_isosurface_mc(self, iso: float = 0, subdivision: int = 31):
        max_v = max(self.values)
        min_v = min(self.values)
        if iso <= min_v or iso >= max_v or (min_v == max_v and min_v == iso):
            return [], []

        step = subdivision * 1j
        x, y, z = np.mgrid[0:1:step, 0:1:step, 0:1:step]
        values = self.trilinear((x, y, z))

        spacing = np.ones(3) / (subdivision - 1)

        # grid = Grid((subdivision, subdivision, subdivision), spacing)
        # vertices, faces = mc_grosso()
        vertices, faces, _, _ = measure.marching_cubes(values, iso, spacing=spacing)
        return vertices, faces
