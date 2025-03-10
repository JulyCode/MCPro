import math

import tables
from grid import *
from renderer import RendererO3D


def _linear_interpolation(p0, p1, interpolation):
    return (1 - interpolation) * p0 + interpolation * p1

def _sq_distance(p0, p1):
    diff = p1 - p0
    return np.dot(diff, diff)

def _cross(p0, p1):
    return np.array([p0[1] * p1[2] - p1[1] * p0[2], p0[2] * p1[0] - p1[2] * p0[0], p0[0] * p1[1] - p1[0] * p0[1]])


# coordinate system
#    w
#    ^  v6--------------v7
#    |  /|             /|
#    | / |     ^v     / |
#    |/  |    /      /  |
#   v4--------------v5  |
#    |   | /        |   |
#    |  v2----------|---v3
#    |  /           |  /
#    | /            | /
#    |/             |/
#   v0--------------v1 --> u


class TMC:
    def __init__(self, grid: Grid, iso: float = 0, create_global_halfedges=True):
        self.grid = grid
        self.iso = iso

        self.vertices = []
        self.faces = []

        self.corner_vertices = {}
        self.edge_vertices = {}
        self.face_vertices = {}
        self.face_segments = {}

        self.create_global_halfedges = create_global_halfedges
        self.global_halfedges = []
        self.global_next = {}
        self.global_opposite = {}
        self.global_face_hes = {}

        self.debug_render = False
        self.singular_counter = 0

    def run(self):
        self.singular_counter = 0
        self.iterate_corners()
        self.iterate_edges()
        self.iterate_faces()
        self.iterate_cells()

        # print(f"singular: {self.singular_counter}")

        return np.array(self.vertices), np.array(self.faces)

    def _add_vertex(self, point):
        self.vertices.append(point)
        return len(self.vertices) - 1


    def iterate_corners(self):
        for idx in self.grid.indices():
            value = self.grid[idx]
            if value == self.iso:
                self.corner_vertices[self.grid.vertex_id(idx)] = self._add_vertex(self.grid.position(idx))


    def iterate_edges(self):
        grid = self.grid
        iso = self.iso

        for v0, v1 in grid.edges():
            f0 = grid[v0]
            f1 = grid[v1]

            if f0 == iso and f1 == iso:
                print("iso edge")
                # edge_segments[grid.edge_id(v0, v1)] = (corner_vertices[grid.vertex_id(v0)], corner_vertices[grid.vertex_id(v1)])
            elif (f0 - iso) * (f1 - iso) < 0:
                p0 = grid.position(v0)
                p1 = grid.position(v1)
                interpolation = (iso - f0) / (f1 - f0)
                p = _linear_interpolation(p0, p1, interpolation)

                self.edge_vertices[grid.edge_id(v0, v1)] = self._add_vertex(p)


    def iterate_faces(self):
        grid = self.grid
        iso = self.iso

        for v0, v1, v2, v3 in grid.faces():
            f0 = grid[v0]
            f1 = grid[v1]
            f2 = grid[v2]
            f3 = grid[v3]

            # determine asymptotic center or replacement if degenerated
            if abs(f0 - f1 - f2 + f3) < 1e-6:
                # bilinear interpolation is a plane, intersection is just a straight line
                asymptotic_center = -1, -1

                if f0 == iso and f1 == iso and f2 == iso and f3 == iso:
                    print('plane case')
                    continue
            else:
                asymptotic_center = (f0 - f2) / (f0 - f1 - f2 + f3), (f0 - f1) / (f0 - f1 - f2 + f3)

            # transform to global
            p0 = grid.position(v0)
            p1 = grid.position(v1)
            p2 = grid.position(v2)
            p3 = grid.position(v3)
            pu0 = _linear_interpolation(p0, p1, asymptotic_center[0])
            pu1 = _linear_interpolation(p2, p3, asymptotic_center[0])
            asymptotic_p = _linear_interpolation(pu0, pu1, asymptotic_center[1])

            # split face in four quadrants
            quadrants = [[], [], [], []]

            # create singular vertex if face is singular and add to all quadrants
            if f0 * f3 - f1 * f2 == 0 and 0 <= asymptotic_center[0] <= 1 and 0 <= asymptotic_center[1] <= 1:  # singular
                self.singular_counter += 1
                # TODO: singular on edge / corner
                new_vertex = self._add_vertex(asymptotic_p)
                self.face_vertices[grid.face_id(v0, v1, v2, v3)] = new_vertex

            # find face local coordinate axes
            compare = []
            if v0[0] == v3[0]:
                compare = [1, 2]
            if v0[1] == v3[1]:
                compare = [2, 0]
            if v0[2] == v3[2]:
                compare = [0, 1]

            # gather all points on this face
            edge_vertices, corner_vertices = self.edge_vertices, self.corner_vertices
            tests = [edge_vertices.get(grid.edge_id(v0, v1)), edge_vertices.get(grid.edge_id(v0, v2)),
                     edge_vertices.get(grid.edge_id(v1, v3)), edge_vertices.get(grid.edge_id(v2, v3)),
                     corner_vertices.get(grid.vertex_id(v0)), corner_vertices.get(grid.vertex_id(v1)),
                     corner_vertices.get(grid.vertex_id(v2)), corner_vertices.get(grid.vertex_id(v3))]

            # sort points into quadrants
            for v in tests:
                if v is None:
                    continue
                vp = self.vertices[v]

                # sorting based on spherical coordinate
                phi = math.atan2(vp[compare[1]] - asymptotic_p[compare[1]], vp[compare[0]] - asymptotic_p[compare[0]]) + math.pi
                idx = int(phi / (math.pi / 2)) % 4
                quadrants[idx].append(v)

            # corner values of the quadrants
            corner_values = [f0, f1, f3, f2]

            segments = []

            # for each quadrant create a segments if it contains two points
            for i, vs in enumerate(quadrants):
                if len(vs) != 2:
                    continue

                vt0, vt1 = vs

                # use x delta in even and y delta in odd quadrants because of singular segments
                normal_axis = compare[i % 2]
                delta = self.vertices[vt1][normal_axis] - self.vertices[vt0][normal_axis]
                # flip normal for lower quadrants
                normal = ((i // 2) * 2 - 1)

                # determine left normal direction based on delta and compare with sign in same corner of asymptotes
                if delta * corner_values[i] * normal > 0:
                    segments.append((vt0, vt1))
                else:
                    segments.append((vt1, vt0))

            # store segments of this face
            self.face_segments[grid.face_id(v0, v1, v2, v3)] = segments


    def iterate_cells(self):
        for idx in self.grid.indices():
            if not (idx[0] < self.grid.size[0] - 1 and idx[1] < self.grid.size[1] - 1 and idx[2] < self.grid.size[2] - 1):
                continue
            # try:
            halfedges, next_halfedge, prev_halfedge, opposite_halfedge, helfedges_per_face = self.process_cell(idx)

            if self.debug_render:
                self.debug_halfedges(self.grid.cell(idx), halfedges, next_halfedge, prev_halfedge, opposite_halfedge)

            if len(halfedges) != 0:
                if self.create_global_halfedges:
                    self.triangulate_global_halfedges(idx, halfedges, next_halfedge, prev_halfedge, opposite_halfedge, helfedges_per_face)
                else:
                    self.triangulate_faces(halfedges, next_halfedge, prev_halfedge)

            # except Exception as e:
            #     print(idx)
            #     print(grid.cell(idx).values)
            #     raise e

    def process_cell(self, idx):
        grid = self.grid
        iso = self.iso
        vertices = self.vertices

        cube = grid.cell(idx)
        values = cube.values

        max_value = max(values)
        min_value = min(values)
        if iso <= min_value or iso >= max_value:  # TODO: plane on face?
            return [], [], [], [], []

        # find saddle points
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
        # if den_u == 0 or den_v == 0:
        #     print(f"linear {det=}")
        if det >= 0  and den_u != 0 and den_v != 0:
            det = math.sqrt(det)
            ui[0] = (num_u - det) / den_u
            ui[1] = (num_u + det) / den_u
            vi[0] = (num_v - det) / den_v
            vi[1] = (num_v + det) / den_v
            # if (e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h) == 0 or (e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h) == 0:
            #     print(values)
            # TODO: find better solution to vertical blue lines? is saddle possible?
            den_w0 = e * ui[0] * vi[0] + f * ui[0] + g * vi[0] + h
            den_w1 = e * ui[1] * vi[1] + f * ui[1] + g * vi[1] + h
            if den_w0 != 0:
                wi[1] = (a * ui[0] * vi[0] + b * ui[0] + c * vi[0] + d) / den_w0
            else:
                wi[1] = 1 if (a * ui[0] * vi[0] + b * ui[0] + c * vi[0] + d) * den_w0 > 0 else 0
            if den_w1 != 0:
                wi[0] = (a * ui[1] * vi[1] + b * ui[1] + c * vi[1] + d) / den_w1
            else:
                wi[0] = 1 if (a * ui[1] * vi[1] + b * ui[1] + c * vi[1] + d) * den_w1 > 0 else 0

            ui[1], ui[0] = ui[0], ui[1]

        inner_points = [(ui[0], vi[0], wi[0]),
                        (ui[0], vi[0], wi[1]),
                        (ui[1], vi[0], wi[1]),
                        (ui[1], vi[1], wi[1]),
                        (ui[1], vi[1], wi[0]),
                        (ui[0], vi[1], wi[0])]

        # collect and orient segments
        halfedges = []
        outgoing_halfedges = {}
        incoming_halfedges = {}
        helfedges_per_face = [[], [], [], [], [], []]

        # build halfedge data structure
        next_halfedge = {}
        prev_halfedge = {}
        opposite_halfedge = {}

        def add_halfedge(v0, v1, v0_singular=False, v1_singular=False):
            # avoid duplicates
            if v0 in outgoing_halfedges:
                for he in outgoing_halfedges[v0]:
                    if halfedges[he][1] == v1:
                        return

            id = len(halfedges)
            halfedges.append((v0, v1))

            if v0 in outgoing_halfedges:
                outgoing_halfedges[v0].append(id)
            else:
                outgoing_halfedges[v0] = [id]

            if v1 in incoming_halfedges:
                incoming_halfedges[v1].append(id)
            else:
                incoming_halfedges[v1] = [id]

            next = outgoing_halfedges.get(v1)
            if next is not None and not v1_singular:
                next_halfedge[id] = next[0]
                prev_halfedge[next[0]] = id

            prev = incoming_halfedges.get(v0)
            if prev is not None and not v0_singular:
                next_halfedge[prev[0]] = id
                prev_halfedge[id] = prev[0]
            return id

        singular_vertex_per_face = [None, None, None, None, None, None]

        # orient halfedges
        face_normal = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
        for face in range(6):
            # global face vertices
            v0, v1, v2, v3 = [(idx[0] + c[0], idx[1] + c[1], idx[2] + c[2]) for c in [cube.corners[i] for i in tables.face_vertices[face]]]

            # compute asymptotes
            f0 = grid[v0]
            f1 = grid[v1]
            f2 = grid[v2]
            f3 = grid[v3]

            if abs(f0 - f1 - f2 + f3) < 1e-6:
                # print('decider not defined')
                asymptotic_center = 0, 0
            else:
                asymptotic_center = (f0 - f2) / (f0 - f1 - f2 + f3), (f0 - f1) / (f0 - f1 - f2 + f3)

            p0 = grid.position(v0)
            p1 = grid.position(v1)
            p2 = grid.position(v2)
            p3 = grid.position(v3)
            pu0 = _linear_interpolation(p0, p1, asymptotic_center[0])
            pu1 = _linear_interpolation(p2, p3, asymptotic_center[0])
            p = _linear_interpolation(pu0, pu1, asymptotic_center[1])

            singular_vertex = self.face_vertices.get(grid.face_id(v0, v1, v2, v3))
            singular_vertex_per_face[face] = singular_vertex
            if singular_vertex is not None:  # singular

                n = face_normal[(face // 2) * 2 + 1]  # according to table, opposing faces have the same normal
                p_local = cube.global_to_local(p)
                h = cube.trilinear((p_local[0] + n[0], p_local[1] + n[1], p_local[2] + n[2]))

                def get_intersection(v0, v1):
                    if grid.edge_id(v0, v1) in self.edge_vertices:
                        return self.edge_vertices[grid.edge_id(v0, v1)]
                    if grid.vertex_id(v0) in self.corner_vertices:
                        return self.corner_vertices[grid.vertex_id(v0)]
                    if grid.vertex_id(v1) in self.corner_vertices:
                        return self.corner_vertices[grid.vertex_id(v1)]
                    print("no intersection found!")

                if h == 0:
                    print("symmetric singular faces")

                # determine direction
                if (f0 < 0 and face % 2 == 0) or (f0 > 0 and face % 2 == 1):
                    he0 = add_halfedge(singular_vertex, get_intersection(v0, v2), v0_singular=True)
                    he1 = add_halfedge(get_intersection(v0, v1), singular_vertex, v1_singular=True)
                    he2 = add_halfedge(singular_vertex, get_intersection(v1, v3), v0_singular=True)
                    he3 = add_halfedge(get_intersection(v2, v3), singular_vertex, v1_singular=True)

                    # determine connectivity
                    if h > 0:
                        next_halfedge[he1] = he0
                        prev_halfedge[he0] = he1
                        next_halfedge[he3] = he2
                        prev_halfedge[he2] = he3
                    else:
                        next_halfedge[he1] = he2
                        prev_halfedge[he2] = he1
                        next_halfedge[he3] = he0
                        prev_halfedge[he0] = he3
                else:
                    he0 = add_halfedge(get_intersection(v0, v2), singular_vertex, v1_singular=True)
                    he1 = add_halfedge(singular_vertex, get_intersection(v0, v1), v0_singular=True)
                    he2 = add_halfedge(get_intersection(v1, v3), singular_vertex, v1_singular=True)
                    he3 = add_halfedge(singular_vertex, get_intersection(v2, v3), v0_singular=True)

                    # determine connectivity
                    if h > 0:
                        next_halfedge[he2] = he1
                        prev_halfedge[he1] = he2
                        next_halfedge[he0] = he3
                        prev_halfedge[he3] = he0
                    else:
                        next_halfedge[he0] = he1
                        prev_halfedge[he1] = he0
                        next_halfedge[he2] = he3
                        prev_halfedge[he3] = he2

                helfedges_per_face[face].append(he0)
                helfedges_per_face[face].append(he1)
                helfedges_per_face[face].append(he2)
                helfedges_per_face[face].append(he3)

            else:
                for sv0, sv1 in self.face_segments[grid.face_id(v0, v1, v2, v3)]:
                    if face % 2 != 0:
                        he = add_halfedge(sv0, sv1)
                    else:
                        he = add_halfedge(sv1, sv0)
                    helfedges_per_face[face].append(he)


        # insert cuts between inner and outer rings
        # compute topology of blue lines
        inner_per_face = [[None, None], [None, None], [None, None], [None, None], [None, None], [None, None]]
        normal_face = [None, None, None, None, None, None]
        if wi[0] < wi[1]:
            inner_per_face[0][1] = 0
            inner_per_face[1][0] = 1
            inner_per_face[1][1] = 3
            inner_per_face[0][0] = 4
        else:
            inner_per_face[0][0] = 1
            inner_per_face[1][1] = 0
            inner_per_face[1][0] = 4
            inner_per_face[0][1] = 3
        if vi[0] < vi[1]:
            inner_per_face[2][1] = 2
            inner_per_face[3][0] = 3
            inner_per_face[3][1] = 5
            inner_per_face[2][0] = 0
        else:
            inner_per_face[2][0] = 3
            inner_per_face[3][1] = 2
            inner_per_face[3][0] = 0
            inner_per_face[2][1] = 5
        if ui[0] < ui[1]:
            inner_per_face[4][1] = 1
            inner_per_face[5][0] = 2
            inner_per_face[5][1] = 4
            inner_per_face[4][0] = 5
        else:
            inner_per_face[4][0] = 2
            inner_per_face[5][1] = 1
            inner_per_face[5][0] = 5
            inner_per_face[4][1] = 4

        if (ui[0] - ui[1]) * (wi[0] - wi[1]) < 0:
            normal_face[1] = 3
            normal_face[4] = 3
        else:
            normal_face[1] = 2
            normal_face[4] = 2
        if (ui[0] - ui[1]) * (vi[0] - vi[1]) < 0:
            normal_face[2] = 1
            normal_face[5] = 1
        else:
            normal_face[2] = 0
            normal_face[5] = 0
        if (vi[0] - vi[1]) * (wi[0] - wi[1]) < 0:
            normal_face[0] = 4
            normal_face[3] = 4
        else:
            normal_face[0] = 5
            normal_face[3] = 5

        start_inner_points = 0
        end_inner_points = 6
        inner_vertices = [None, None, None, None, None, None]
        inner_singular = [False] * 6
        inner_used = [True] * 6
        for i, p in enumerate(inner_points):
            if not (-0.000001 <= p[0] <= 1.000001 and -0.000001 <= p[1] <= 1.000001 and -0.000001 <= p[2] <= 1.000001):
                if inner_used[i - 1]:
                    end_inner_points = i
                inner_used[i] = False
                continue
            else:
                if not inner_used[i - 1]:
                    start_inner_points = i
                inner_used[i] = True

            axis = 2 - normal_face[i] // 2
            face_offset = 0 if p[axis] < 0.5 else 1
            corresponding_face = (normal_face[i] // 2) * 2 + face_offset
            singular_vertex = singular_vertex_per_face[corresponding_face]
            choose_other_vertex = abs(inner_points[(i + 3) % 6][axis] - face_offset) < abs(p[axis] - face_offset)  # check if the other vertex is closer to the face
            if singular_vertex is None or choose_other_vertex:
                v = len(vertices)
                inner_vertices[i] = v
                p = min(max(p[0], 0.05), 0.95), min(max(p[1], 0.05), 0.95), min(max(p[2], 0.05), 0.95)
                vertices.append(cube.local_to_global(p))
            else:
                inner_singular[i] = True
                inner_vertices[i] = singular_vertex

        # find two closest contour vertices connected with blue lines
        outer_halfedges = [[None, None], [None, None], [None, None], [None, None], [None, None], [None, None]]
        outer_halfedges_vertex = [[None, None], [None, None], [None, None], [None, None], [None, None], [None, None]]
        for face in range(6):
            singular_vertex = singular_vertex_per_face[face]

            for he in helfedges_per_face[face]:
                p0 = vertices[halfedges[he][0]]
                p1 = vertices[halfedges[he][1]]

                inner0, inner1 = inner_per_face[face]

                v0, v1, v2, v3 = [(idx[0] + c[0], idx[1] + c[1], idx[2] + c[2]) for c in
                                  [cube.corners[i] for i in tables.face_vertices[face]]]
                f0 = grid[v0]
                f1 = grid[v1]
                f2 = grid[v2]
                f3 = grid[v3]

                if abs(f0 - f1 - f2 + f3) < 1e-6:
                    # print('decider not defined')
                    asymptotic_center = -1, -1
                else:
                    asymptotic_center = (f0 - f1) / (f0 - f1 - f2 + f3), (f0 - f2) / (f0 - f1 - f2 + f3)

                c0 = grid.position(v0)
                c1 = grid.position(v1)
                c2 = grid.position(v2)
                c3 = grid.position(v3)
                pu0 = _linear_interpolation(c0, c1, asymptotic_center[0])
                pu1 = _linear_interpolation(c2, c3, asymptotic_center[0])
                asymp_p = _linear_interpolation(pu0, pu1, asymptotic_center[1])

                he_dist0 = _sq_distance(asymp_p, p0)
                he_dist1 = _sq_distance(asymp_p, p1)
                he_test_p = p0 if he_dist0 > he_dist1 else p1

                face_max = max(c0[0], c1[0], c2[0], c3[0]), max(c0[1], c1[1], c2[1], c3[1]), max(c0[2], c1[2], c2[2], c3[2])
                face_min = min(c0[0], c1[0], c2[0], c3[0]), min(c0[1], c1[1], c2[1], c3[1]), min(c0[2], c1[2], c2[2], c3[2])

                # check if this halfedge is connected to the inner point
                p = inner_points[inner0]
                p = cube.local_to_global(p)
                p = min(max(p[0], face_min[0]), face_max[0]), min(max(p[1], face_min[1]), face_max[1]), min(max(p[2], face_min[2]), face_max[2])
                inner0_hit = False
                if min(p0[0], p1[0]) - p[0] < 1e-6 and max(p0[0], p1[0]) - p[0] > -1e-6 and \
                        min(p0[1], p1[1]) - p[1] < 1e-6 and max(p0[1], p1[1]) - p[1] > -1e-6 and \
                        min(p0[2], p1[2]) - p[2] < 1e-6 and max(p0[2], p1[2]) - p[2] > -1e-6:
                    inner0_hit = True
                    inner0_l0 = _sq_distance(p, p0)
                    inner0_l1 = _sq_distance(p, p1)

                p = inner_points[inner1]
                p = cube.local_to_global(p)
                p = min(max(p[0], face_min[0]), face_max[0]), min(max(p[1], face_min[1]), face_max[1]), min(max(p[2], face_min[2]), face_max[2])
                inner1_hit = False
                if min(p0[0], p1[0]) - p[0] < 1e-6 and max(p0[0], p1[0]) - p[0] > -1e-6 and \
                        min(p0[1], p1[1]) - p[1] < 1e-6 and max(p0[1], p1[1]) - p[1] > -1e-6 and \
                        min(p0[2], p1[2]) - p[2] < 1e-6 and max(p0[2], p1[2]) - p[2] > -1e-6:
                    inner1_hit = True
                    inner1_l0 = _sq_distance(p, p0)
                    inner1_l1 = _sq_distance(p, p1)

                # exception for singular, always take center
                if singular_vertex is not None:
                    if singular_vertex == halfedges[he][0]:
                        inner0_l0 = 0
                        inner1_l0 = 0
                    if singular_vertex == halfedges[he][1]:
                        inner0_l1 = 0
                        inner1_l1 = 0

                # choose closer vertex of halfedge
                if inner0_hit and inner1_hit:
                    if inner0_l0 < inner1_l0:
                        outer_halfedges[inner0][1] = he
                        outer_halfedges_vertex[inner0][1] = 0
                        outer_halfedges[inner1][0] = he
                        outer_halfedges_vertex[inner1][0] = 1
                    else:
                        outer_halfedges[inner0][1] = he
                        outer_halfedges_vertex[inner0][1] = 1
                        outer_halfedges[inner1][0] = he
                        outer_halfedges_vertex[inner1][0] = 0
                elif inner0_hit:
                    outer_halfedges[inner0][1] = he
                    outer_halfedges_vertex[inner0][1] = 0 if inner0_l0 < inner0_l1 else 1
                elif inner1_hit:
                    outer_halfedges[inner1][0] = he
                    outer_halfedges_vertex[inner1][0] = 0 if inner1_l0 < inner1_l1 else 1

        # iterate over inner hexagon to create connections
        last_halfedges = None
        first_halfedges = None
        i = start_inner_points
        while i != end_inner_points:
            # skip singular faces
            if inner_singular[i]:
                if i == 5 and end_inner_points == 6:
                    break
                i = (i + 1) % 6
                continue

            v = inner_vertices[i]

            incoming_candidates = []
            outgoing_candidates = []

            # halfedges between this and previous inner point
            if start_inner_points == 0 and end_inner_points == 6:
                if i == 0  and not inner_singular[(i - 1) % 6]:
                    first_halfedges = (len(halfedges), len(halfedges) + 1)
                    opposite_halfedge[len(halfedges)] = len(halfedges) + 1
                    opposite_halfedge[len(halfedges) + 1] = len(halfedges)
                    halfedges.append((inner_vertices[(i - 1) % 6], v))
                    halfedges.append((v, inner_vertices[(i - 1) % 6]))
                    last_halfedges = first_halfedges

            # remove shoulders for singular points
            if inner_singular[i]:
                outer_halfedges[i].clear()
                outer_halfedges_vertex[i].clear()
            else:
                # add beginning halfedges, if prev is outside
                if inner_used[i - 1]:
                    if last_halfedges is not None:
                        incoming_candidates.append(last_halfedges[0])
                        outgoing_candidates.append(last_halfedges[1])
                else:
                    outer_halfedges[i].insert(0, outer_halfedges[i - 1][0])
                    outer_halfedges_vertex[i].insert(0, outer_halfedges_vertex[i - 1][0])

                # add end halfedges, if next is outside
                if not inner_used[(i + 1) % 6]:
                    outer_halfedges[i].append(outer_halfedges[(i + 1) % 6][1])
                    outer_halfedges_vertex[i].append(outer_halfedges_vertex[(i + 1) % 6][1])

            # compute outer vertices
            outer_vertices = [halfedges[outer_halfedges[i][j]][outer_halfedges_vertex[i][j]] if outer_halfedges[i][j] is not None else None for j in
                              range(len(outer_halfedges[i]))]

            # remove duplicated outer vertices
            del_offset = 0
            for j in range(1, len(outer_halfedges[i]) + 1):
                j_idx = j - del_offset - 1
                j_next_idx = (j - del_offset) % len(outer_halfedges[i])
                if j_idx != j_next_idx and outer_vertices[j_idx] == outer_vertices[j_next_idx]:
                    # print("same outer vertices")
                    outer_vertices.pop(j_next_idx)
                    outer_halfedges[i].pop(j_next_idx)
                    outer_halfedges_vertex[i].pop(j_next_idx)
                    del_offset += 1

            # halfedges from inner point to outer contour
            for j, other_he in enumerate(outer_halfedges[i]):
                if outer_vertices[j] is None:
                    continue

                from_outer_he = len(halfedges)
                halfedges.append((outer_vertices[j], v))
                to_outer_he = len(halfedges)
                halfedges.append((v, outer_vertices[j]))

                opposite_halfedge[from_outer_he] = to_outer_he
                opposite_halfedge[to_outer_he] = from_outer_he

                outer_prev = other_he if outer_halfedges_vertex[i][j] == 1 else prev_halfedge[other_he]
                outer_next = other_he if outer_halfedges_vertex[i][j] == 0 else next_halfedge[other_he]

                if outer_prev is None:
                    print("no outer prev")
                next_halfedge[outer_prev] = from_outer_he
                prev_halfedge[from_outer_he] = outer_prev

                if outer_next is None:
                    print("no outer next")
                prev_halfedge[outer_next] = to_outer_he
                next_halfedge[to_outer_he] = outer_next

                incoming_candidates.append(from_outer_he)
                outgoing_candidates.append(to_outer_he)

            # halfedges between this and next inner point
            if inner_used[(i + 1) % 6] and not inner_singular[(i + 1) % 6]:
                if i != end_inner_points - 1:
                    next_inner = inner_vertices[(i + 1) % 6]
                    last_halfedges = len(halfedges), len(halfedges) + 1
                    opposite_halfedge[len(halfedges)] = len(halfedges) + 1
                    opposite_halfedge[len(halfedges) + 1] = len(halfedges)
                    halfedges.append((v, next_inner))
                    halfedges.append((next_inner, v))

                    incoming_candidates.append(last_halfedges[1])
                    outgoing_candidates.append(last_halfedges[0])
                else:
                    incoming_candidates.append(first_halfedges[1])
                    outgoing_candidates.append(first_halfedges[0])
            else:
                last_halfedges = None

            # compute orientation
            inner_p = inner_points[i]
            n = face_normal[normal_face[i]]
            h = cube.trilinear((inner_p[0] + n[0], inner_p[1] + n[1], inner_p[2] + n[2]))
            sign = 1 if h > 0 else -1  # TODO: case h == 0

            if sign > 0:
                for k in range(0, len(incoming_candidates), 1):
                    inc = k
                    out = (k + 1) % len(incoming_candidates)
                    if incoming_candidates[inc] == incoming_candidates[out]:
                        continue

                    next_halfedge[incoming_candidates[inc]] = outgoing_candidates[out]
                    prev_halfedge[outgoing_candidates[out]] = incoming_candidates[inc]
            else:
                for k in range(len(incoming_candidates) - 1, -1, -1):
                    inc = k
                    out = (k - 1) % len(incoming_candidates)
                    if incoming_candidates[inc] == incoming_candidates[out]:
                        continue

                    next_halfedge[incoming_candidates[inc]] = outgoing_candidates[out]
                    prev_halfedge[outgoing_candidates[out]] = incoming_candidates[inc]

            if i == 5 and end_inner_points == 6:
                break
            i = (i + 1) % 6

        for i, he in enumerate(halfedges):
            if i in next_halfedge and next_halfedge[i] in next_halfedge and halfedges[next_halfedge[i]][1] == he[0]:
                next_halfedge[prev_halfedge[i]] = next_halfedge[next_halfedge[i]]
                prev_halfedge[next_halfedge[next_halfedge[i]]] = prev_halfedge[i]
                prev_halfedge[i] = None
                next_halfedge[next_halfedge[i]] = None

        return halfedges, next_halfedge, prev_halfedge, opposite_halfedge, helfedges_per_face


    def triangulate_faces(self, halfedges, next_halfedge, prev_halfedge):
        # contour-free version
        for i, he in enumerate(halfedges):
            v1, v2 = he
            prev_he = prev_halfedge.get(i)
            if prev_he is None:
                print("no previous halfedge")
                continue
            v0 = halfedges[prev_he][0]
            if v0 == v2:  # opposite halfedges
                prev_halfedge[prev_he] = None
            else:
                # create face and remove this halfedge
                self.faces.append([v0, v1, v2])
                halfedges[prev_he] = (v0, v2)

                next_he = next_halfedge.get(i)
                if next_he is None:
                    print("no next halfedge")
                    continue
                next_halfedge[prev_he] = next_he
                prev_halfedge[next_he] = prev_he


    def triangulate_global_halfedges(self, idx, halfedges, next_halfedge, prev_halfedge, opposite_halfedge, helfedges_per_face):
        # explicit contour version
        contours = []
        visited = set()
        for i, he in enumerate(halfedges):
            if i in visited:
                continue
            contour = [i]
            visited.add(i)
            next_he = next_halfedge.get(i)
            while next_he is not None and next_he != i:
                if next_he in visited:
                    # print("loop")
                    break
                contour.append(next_he)
                visited.add(next_he)
                next_he = next_halfedge.get(next_he)
            if next_he == i:
                contours.append(contour)

        faces_he = []  # list of faces, represented by one halfedge index of the face
        he_to_face = {}
        for contour in contours:
            v0 = halfedges[contour[0]][0]
            for i in range(1, len(contour) - 2):
                v1, v2 = halfedges[contour[i]]

                he0 = len(halfedges)
                halfedges.append((v2, v0))
                he1 = len(halfedges)
                halfedges.append((v0, v2))

                opposite_halfedge[he0] = he1
                opposite_halfedge[he1] = he0

                prev_halfedge[next_halfedge[contour[i]]] = he1
                next_halfedge[he1] = next_halfedge[contour[i]]
                next_halfedge[prev_halfedge[prev_halfedge[contour[i]]]] = he1
                prev_halfedge[he1] = prev_halfedge[prev_halfedge[contour[i]]]

                next_halfedge[contour[i]] = he0
                prev_halfedge[he0] = contour[i]
                next_halfedge[he0] = prev_halfedge[contour[i]]
                prev_halfedge[prev_halfedge[contour[i]]] = he0

                # create face
                faces_he.append(contour[i])
                he_to_face[contour[i]] = len(faces_he) - 1
                he_to_face[he0] = len(faces_he) - 1
                he_to_face[prev_halfedge[contour[i]]] = len(faces_he) - 1

            faces_he.append(contour[-2])
            he_to_face[contour[-2]] = len(faces_he) - 1
            he_to_face[contour[-1]] = len(faces_he) - 1
            he_to_face[prev_halfedge[contour[-2]]] = len(faces_he) - 1

        # flip triangulation of two faces that share three vertices but only two edges
        # for he in faces_he:
        #     # get vertices of face
        #     vts = [halfedges[he][0], halfedges[he][1], halfedges[next_halfedge[he]][1]]
        #     for i in range(3):
        #         if he not in opposite_halfedge:
        #             continue
        #         opp = opposite_halfedge[he]
        #         opp_vertex = halfedges[next_halfedge[opp]][1]
        #         if opp_vertex in vts:
        #             # duplicate face found
        #             # find the open edge
        #             he = next_halfedge[he]
        #             if he not in opposite_halfedge or he_to_face[opposite_halfedge[he]] == he_to_face[opp]:
        #                 he = next_halfedge[he]
        #             if he_to_face[opposite_halfedge[he]] == he_to_face[opp]:
        #                 print("still wrong")
        #             opp = opposite_halfedge[he]
        #             opp_vertex = halfedges[next_halfedge[opp]][1]
        #             # change top vertex
        #             prev_halfedge[next_halfedge[he]] = prev_halfedge[opp]
        #             next_halfedge[prev_halfedge[opp]] = next_halfedge[he]
        #             # change bottom vertex
        #             prev_halfedge[next_halfedge[opp]] = prev_halfedge[he]
        #             next_halfedge[prev_halfedge[he]] = next_halfedge[opp]
        #             # flip halfedges
        #             halfedges[he] = halfedges[next_halfedge[he]][1], opp_vertex
        #             halfedges[opp] = opp_vertex, halfedges[next_halfedge[he]][1]
        #             # change left vertex
        #             next_halfedge[next_halfedge[he]] = he
        #             prev_halfedge[prev_halfedge[he]] = opp
        #             # change right vertex
        #             next_halfedge[next_halfedge[opp]] = opp
        #             prev_halfedge[prev_halfedge[opp]] = he
        #             # change middle halfedges
        #             prev_halfedge[he] = next_halfedge[he]
        #             next_halfedge[he] = prev_halfedge[opp]
        #             prev_halfedge[opp] = next_halfedge[opp]
        #             next_halfedge[opp] = prev_halfedge[prev_halfedge[opp]]
        #             # update face handle
        #             faces_he[he_to_face[he]] = he
        #             faces_he[he_to_face[opp]] = opp
        #             he_to_face[next_halfedge[he]] = he_to_face[he]
        #             he_to_face[next_halfedge[opp]] = he_to_face[opp]
        #             break
        #
        #         he = next_halfedge[he]

        for he in faces_he:
            self.faces.append([halfedges[he][0], halfedges[he][1], halfedges[next_halfedge[he]][1]])

        # create global halfedge data structure
        used_halfedges = []
        translation_to_new = {}
        translation_to_old = {}
        for he in range(len(halfedges)):
            if next_halfedge.get(he) is not None and prev_halfedge.get(he) is not None:
                translation_to_old[len(used_halfedges)] = he
                translation_to_new[he] = len(used_halfedges)
                used_halfedges.append(halfedges[he])

        he_offset = len(self.global_halfedges)
        self.global_halfedges += used_halfedges

        # set global next pointers
        for he in range(len(used_halfedges)):
            next_he = translation_to_new[next_halfedge.get(translation_to_old[he])]
            self.global_next[he + he_offset] = next_he + he_offset

        # set global opp pointers for inner halfedges (within one cell)
        for he in range(len(used_halfedges)):
            opp = opposite_halfedge.get(translation_to_old[he])
            if opp is not None:
                self.global_opposite[he + he_offset] = translation_to_new[opp] + he_offset

        # set opp pointers between cells
        cube = self.grid.cell(idx)
        for face in range(6):
            # global face vertices
            v0, v1, v2, v3 = [(idx[0] + c[0], idx[1] + c[1], idx[2] + c[2]) for c in
                              [cube.corners[i] for i in tables.face_vertices[face]]]

            face_idx = self.grid.face_id(v0, v1, v2, v3)
            face_hes = self.global_face_hes.get(face_idx)
            if face_hes is None:
                self.global_face_hes[face_idx] = [translation_to_new[he] + he_offset for he in helfedges_per_face[face]]
            else:
                for he in face_hes:
                    vts0 = tuple(sorted(self.global_halfedges[he]))
                    for opp_he in helfedges_per_face[face]:
                        opp_he = translation_to_new[opp_he]
                        vts1 = tuple(sorted(self.global_halfedges[opp_he + he_offset]))
                        if vts0 == vts1:
                            self.global_opposite[opp_he + he_offset] = he
                            self.global_opposite[he] = opp_he + he_offset


    def debug_halfedges(self, cube, halfedges, next_halfedge, prev_halfedge, opposite_halfedge):
        vertices = self.vertices

        renderer = RendererO3D()

        print(halfedges)
        print(next_halfedge)

        def show_halfedges():
            global he_idx
            if he_idx >= len(halfedges):
                he_idx = 0
            if he_idx < 0:
                he_idx = len(halfedges) - 1
            renderer.clear()
            renderer.render_cube(cube)
            renderer.render_blue_lines(cube)
            renderer.render_corners(cube)
            he_points = []
            for i, he in enumerate(halfedges):
                if i == he_idx or opposite_halfedge.get(i) == he_idx:
                    continue
                he_points.append(vertices[he[0]])
                he_points.append(vertices[he[1]])
            renderer.render_lines("halfedges", he_points, color="green")
            renderer.render_halfedge("highlight", vertices[halfedges[he_idx][0]], vertices[halfedges[he_idx][1]])
            print(he_idx)

        def dec_he(vis, action, mods):
            if action != 0:
                return
            global he_idx
            he_idx -= 1
            show_halfedges()

        def inc_he(vis, action, mods):
            if action != 0:
                return
            global he_idx
            he_idx += 1
            show_halfedges()

        def find_next_he(vis, action, mods):
            if action != 0:
                return
            global he_idx
            next_he = next_halfedge.get(he_idx)
            if next_he is not None:
                he_idx = next_he
            show_halfedges()

        def find_prev_he(vis, action, mods):
            if action != 0:
                return
            global he_idx
            prev_he = prev_halfedge.get(he_idx)
            if prev_he is not None:
                he_idx = prev_he
            show_halfedges()

        renderer.render_cube(cube)
        renderer.render_blue_lines(cube)
        renderer.key_callbacks(
            [("left", 263, dec_he), ("right", 262, inc_he), ("up", 265, find_prev_he), ("down", 264, find_next_he)])
        renderer.show()
        renderer.run()
        del renderer

he_idx = 0

def tmc(grid: Grid, iso: float = 0):
    if iso != 0:
        for idx in grid.indices():
            grid[idx] -= iso
        iso = 0

    mc = TMC(grid, iso)
    return mc.run()


def tmc_halfedges(grid: Grid, iso: float = 0):
    if iso != 0:
        for idx in grid.indices():
            grid[idx] -= iso
        iso = 0

    mc = TMC(grid, iso, create_global_halfedges=True)
    vertices, faces =  mc.run()

    return np.array(vertices), np.array(faces), mc.global_halfedges, mc.global_next, mc.global_opposite
