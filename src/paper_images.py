
from sympy.physics.control.control_plots import matplotlib

import mc

from comparisons import *
from renderer import *
from tests import *

class Visualizations:
    @staticmethod
    def hyperbola():
        return [-1, 2, -1, 2, 3, -4, 3, -4]

    @staticmethod
    def singular():
        return [-3, 1, -3, 1, 6, -2, 6, -2]

    @staticmethod
    def linear():
        return [-1, -3, -1, -3, 4, 2, 4, 2]

    @staticmethod
    def plane():
        return [0, 0, -1, -3, 0, 0, -4, -2]

    @staticmethod
    def ambiguous_face_separated():
        return [-2, 2.2, 8, 8, 2, -1.9, 8, 8]

    @staticmethod
    def ambiguous_face_connected():
        return [-2.2, 2, 8, 8, 1.9, -2, 8, 8]

    @staticmethod
    def mc4_no_tunnel():
        return [-2.95, 1, 1, 1, 1, 1, 1, -2.95]

    @staticmethod
    def mc4_with_tunnel():
        return [-3.05, 1, 1, 1, 1, 1, 1, -3.05]


def reference_trilinear(grid, subdivision=101):
    total_vertices = []
    total_faces = []
    for cell in grid.cells():
        offset = len(total_vertices)
        vertices, faces = cell.trilinear_isosurface_mc(subdivision=subdivision)
        for v in vertices:
            total_vertices.append(cell.local_to_global(v))
        for v0, v1, v2 in faces:
            total_faces.append((v0 + offset, v1 + offset, v2 + offset))
    return total_vertices, total_faces

def face_intersection(cube):
    f0, f1, f2, f3 = cube.values[0], cube.values[1], cube.values[4], cube.values[5]
    c0, c1, c2, c3 = np.array(cube.corners[0], dtype=np.float64), np.array(cube.corners[1], dtype=np.float64), np.array(cube.corners[4], dtype=np.float64), np.array(cube.corners[5], dtype=np.float64)

    points = []

    if abs(f0 - f1 - f2 + f3) < 1e-6:
        if f0 - f2 > f0 - f1:
            u0 = (f0 - 0 * (f0 - f2)) / (f0 - f1)
            u1 = (f0 - 1 * (f0 - f2)) / (f0 - f1)
            points.append((u0, 0))
            points.append((u1, 1))
        else:
            v0 = (f0 - 0 * (f0 - f1)) / (f0 - f2)
            v1 = (f0 - 1 * (f0 - f1)) / (f0 - f2)
            points.append((0, v0))
            points.append((1, v1))
    else:
        asymptotic_center = (f0 - f1) / (f0 - f1 - f2 + f3), (f0 - f2) / (f0 - f1 - f2 + f3)
        if f0 * f3 - f1 * f2 == 0:
            points.append((asymptotic_center[0], 0))
            points.append((asymptotic_center[0], 1))
            points.append((0, asymptotic_center[1]))
            points.append((1, asymptotic_center[1]))
        else:
            v = np.linspace(0, 1, 5000)

            u = (-f0 + v * (f0 - f2)) / (-f0 + f1 + v * (f0 - f1 - f2 + f3))
            for i in range(v.shape[0] - 1):
                # if -1 <= u[i] <= 0:
                #     u[i] = 0
                # if 1 <= u[i] <= 2:
                #     u[i] = 1
                # if -1 <= u[i + 1] <= 0:
                #     u[i + 1] = 0
                # if 1 <= u[i + 1] <= 2:
                #     u[i + 1] = 1
                if not math.isinf(u[i]) and not math.isinf(u[i + 1]) and 0 <= u[i] <= 1 and 0 <= u[i + 1] <= 1:
                    points.append((u[i], v[i]))
                    points.append((u[i + 1], v[i + 1]))

    points = np.array(points)
    return [(p[0], p[1] + 0.01, p[2]) for p in [cube.local_to_global(mc._linear_interpolation(mc._linear_interpolation(c0, c1, p[0]), mc._linear_interpolation(c2, c3, p[0]), p[1])) for p in points]]

def asymptotes(cube):
    f0, f2, f1, f3 = cube.values[0], cube.values[1], cube.values[4], cube.values[5]
    c0, c1, c2, c3 = np.array(cube.corners[0], dtype=np.float64), np.array(cube.corners[1], dtype=np.float64), np.array(cube.corners[4], dtype=np.float64), np.array(cube.corners[5], dtype=np.float64)

    asymptotic_center = (f0 - f1) / (f0 - f1 - f2 + f3), (f0 - f2) / (f0 - f1 - f2 + f3)
    points = [(0, asymptotic_center[1]), (1, asymptotic_center[1]), (asymptotic_center[0], 0), (asymptotic_center[0], 1)]
    return [(p[0], p[1] + 0.01, p[2]) for p in [cube.local_to_global(mc._linear_interpolation(mc._linear_interpolation(c0, c1, p[0]), mc._linear_interpolation(c2, c3, p[0]), p[1])) for p in points]]

def contours(grid):
    tmc = TMC(grid, 0)

    points = []
    for v0, v1 in tmc.face_segments:
        points.append(tmc.vertices[v0])
        points.append(tmc.vertices[v1])
    return points

def to_grid(values):
    grid = Grid((2, 2, 2), (1, 1, 1))
    grid.values = values
    return grid

def to_viewpoint(front):
    return (np.array(front) - np.ones(3) * 0) * 500 + np.ones(3) * 0

def render_comparison(grid, cam, name):
    compositions = []
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}_trilinear.png", show_cubes=True,
                    show_blue_lines=True))
    compositions.append(
        Composition(tmc(grid), grid, cam, f"images/{name}_ours.png", show_cubes=True, show_blue_lines=True, show_edges=True, duplicate_vertices=True))
    compositions.append(
        Composition(mc_skimage(grid), grid, cam, f"images/{name}_mc33.png", show_cubes=True, show_blue_lines=False, show_edges=True, duplicate_vertices=True))
    return compositions

class Lines:
    def __init__(self, points, width=4, color=(190/320, 33/320, 265/320), cylinders=False):
        self.points = points
        self.width = width
        self.color = color
        self.cylinders = cylinders

class Composition:
    def __init__(self, mesh, grid, camera, file, center=None, lines = None, halfedges=False, show_edges=False, duplicate_vertices=False, show_cubes=False, show_blue_lines=False, show_hexagon=False, show_asymptotes=False, show_mesh=True, show_corners=True):
        self.mesh = mesh
        self.lines = [lines] if isinstance(lines, Lines) else lines
        self.halfedges = halfedges
        self.grid = grid
        self.camera = camera
        self.center = center
        self.file = file
        self.show_edges = show_edges
        self.duplicate_vertices = duplicate_vertices
        self.show_cubes = show_cubes
        self.show_blue_lines = show_blue_lines
        self.show_hexagon = show_hexagon
        self.show_asymptotes = show_asymptotes
        self.show_mesh = show_mesh
        self.show_corners = show_corners

def main():
    renderer = RendererO3D(resolution=(3840*2, 2160*2), legacy=False, zoom=1.9, transparent_bg=True)
    renderer.show()
    compositions = []

    fast = True

    ##### figures #####
    cam = to_viewpoint([0.25696341815520857, 0.31309725285023604, 0.91429749643517388])
    grid = to_grid(Visualizations.mc4_no_tunnel())
    name = "mc4_no_tunnel"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_edges=False))

    grid = to_grid(Visualizations.mc4_with_tunnel())
    name = "mc4_with_tunnel"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_edges=False))

    cam = to_viewpoint([-0.13838152455498889, 0.75025829572936464, 0.6465006135736503])
    grid = to_grid(example_single_saddle_6_vts())
    name = "saddle_point"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=True, show_edges=False))

    cam = to_viewpoint([-0.46789646193827888, 0.43727308822940997, 0.76802678808486591])
    grid = to_grid(example_MC_4_with_tunnel())
    name = "hexagon"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=True, show_hexagon=True, show_mesh=False))

    vertices = np.array([(0.6603073271384561, 0.0, 0.0), (0.0, 0.2653515072018509, 0.0), (0.0, 0.0, 0.41912865230682056),
     (0.5947149055526809, 0.0, 1.0), (0.0, 0.2598655514694427, 1.0), (0.8659146529279096, 1.0, 0.0),
     (0.0, 1.0, 0.4122474873801812), (0.6460453241311952, 1.0, 1.0), (1.0, 0.5454514352001958, 0.0),
     (1.0, 0.0, 0.3526241987803729), (1.0, 0.30397120363436575, 1.0), (1.0, 1.0, 0.16544067787729339),
     [0.5590933, 0.13513676, 0.64869556], [0.5590933, 0.13513676, 0.4245313],
     [0.10942802, 0.13513676, 0.4245313], [0.10942802, 0.25646476, 0.4245313],
     [0.10942802, 0.25646476, 0.64869556], [0.5590933, 0.25646476, 0.64869556]])
    halfedges_contour = [(8, 5), (1, 0), (10, 7), (4, 3), (3, 2), (0, 9), (7, 6), (5, 11), (2, 4), (6, 1), (9, 10), (11, 8)]
    # halfedges_inner = [(17, 12), (12, 17), (3, 12), (12, 3), (12, 13), (13, 12), (9, 13), (13, 9), (0, 13), (13, 0), (13, 14), (14, 13), (2, 14), (14, 2), (14, 15), (15, 14), (1, 15), (15, 1), (6, 15), (15, 6), (15, 16), (16, 15), (4, 16), (16, 4), (16, 17), (17, 16), (7, 17), (17, 7), (10, 17), (17, 10)]
    halfedges_inner = [(3, 12), (12, 3), (9, 13), (13, 9), (0, 13), (13, 0),
                       (2, 14), (14, 2), (1, 15), (15, 1), (6, 15), (15, 6),
                       (4, 16), (16, 4), (7, 17), (17, 7), (10, 17), (17, 10)]

    lines_contour = []
    for he in halfedges_contour:
        lines_contour.append(vertices[he[0]])
        lines_contour.append(vertices[he[1]])
    lines_inner = []
    for he in halfedges_inner:
        lines_inner.append(vertices[he[0]])
        lines_inner.append(vertices[he[1]])

    # cam = to_viewpoint([0.55206138887542744, 0.44226402124762992, 0.70684563974237702])
    # cam = to_viewpoint([0.6560808966985705, 0.51433981865187373, 0.552279284362864])
    # cam = to_viewpoint([0.30903424189667938, 0.6214873824438959, 0.71989670842307618])
    cam = to_viewpoint([0.22198612188449593, 0.55432444880051679, 0.80215121214935847])

    grid = to_grid(example_MC_13_with_tunnel())
    name = "contours"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_mesh=False, lines=Lines(lines_contour, cylinders=True)))

    name = "inner_skeleton"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_mesh=False, lines=Lines(lines_inner, width=4, cylinders=True), show_hexagon=True))

    halfedges = [(8, 5), (1, 0), (10, 7), (4, 3), (3, 2), (0, 9), (7, 6), (5, 11), (2, 4), (6, 1), (9, 10), (11, 8), (17, 12), (12, 17), (3, 12), (12, 3), (12, 13), (13, 12), (9, 13), (13, 9), (0, 13), (13, 0),
                       (13, 14), (14, 13), (2, 14), (14, 2), (14, 15), (15, 14), (1, 15), (15, 1), (6, 15), (15, 6),
                       (15, 16), (16, 15), (4, 16), (16, 4), (16, 17), (17, 16), (7, 17), (17, 7), (10, 17), (17, 10)]
    contours = [[0, 7, 11], [1, 20, 22, 26, 29], [2, 38, 41], [3, 14, 13, 37, 35], [4, 24, 23, 17, 15], [5, 18, 21],
     [6, 30, 32, 36, 39], [8, 34, 33, 27, 25], [9, 28, 31], [10, 40, 12, 16, 19]]
    colors = np.array(matplotlib.colormaps["tab10"].colors)
    contours_lines = []
    for i, contour in enumerate(contours):
        points = []
        for he in contour:
            points.append(vertices[halfedges[he][0]])
            points.append(vertices[halfedges[he][1]])
        contours_lines.append(Lines(points, color=colors[i], cylinders=True))

    name = "polygons"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_mesh=False, lines=contours_lines))

    name = "triangulated_mesh"
    compositions.append(
        Composition(tmc(grid), grid, cam, f"images/{name}.png", show_cubes=True, duplicate_vertices=True))

    cam = to_viewpoint(np.array([0.5725065421143184, 0.36402031176681915, 0.67466010634680168]) * 1.5)
    grid = grid2_simple_singular()
    name = "grid2_simple_singular"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=True))

    grid = grid2_non_manifold_singular()
    name = "grid2_non_manifold_singular"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=True))

    cam = to_viewpoint([0.90819534807000168, 0.14028368496548732, 0.39433703538535714])
    grid = grid2_simple_ambiguous()
    vertices = [[0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 1], [0, 1, 0.5], [1, 0, 0.6], [1, 0.6, 0], [1, 0.4, 1], [1, 1, 0.4], [2, 0, 0.5], [2, 0.5, 0], [2, 0.5, 1], [2, 1, 0.5]]
    faces = [[0, 1, 4], [4, 1, 5], [3, 2, 6], [6, 7, 3], [4, 6, 8], [8, 6, 10], [5, 7, 9], [9, 7, 11]]
    name = "grid2_simple_ambiguous"
    compositions.append(
        Composition((vertices, faces), grid, cam, f"images/{name}.png", show_cubes=True, duplicate_vertices=True))

    grid = to_grid(example_MC_4_with_tunnel())
    cam = to_viewpoint([0.17114801521845338, 0.41144995732052159, 0.89521912932405834])
    name = "mc4_tunnel_tri"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True))

    name = "mc4_tunnel_mc"
    compositions.append(
        Composition(mc_grosso(grid), grid, cam, f"images/{name}.png", show_cubes=True))

    cam = to_viewpoint(np.array([-0.35498152203944255, 0.78764588506633804, 0.50358919641770117]) * 1.1)
    grid = to_grid(example_cross())
    name = "cross"
    compositions.append(
        Composition(reference_trilinear(grid, 99 if fast else 666), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    cam = to_viewpoint([-0.56354291273771362, 0.31377939241446101, -0.76417398437731798])
    grid = to_grid(example_intersecting_planes())
    name = "intersecting_planes"
    compositions.append(
        Composition(reference_trilinear(grid, 99 if fast else 666), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    cam = to_viewpoint(np.array([0.54689488368520367, 0.54813467757458068, -0.63281463434338991]) * 1.05)
    grid = to_grid(example_degenerated_tunnel())
    name = "degenerated_tunnel"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    # name = "degenerated_tunnel_ours"
    # compositions.append(
    #     Composition(tmc(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))  # TODO: use?

    # cam = to_viewpoint(np.array([-0.62173874488329606, 0.19458783356196771, 0.75866758737974549]) * 1)
    # cam = np.array([-111100.57, 21597.18, 150056.28])
    # cam = [-0.62173874488329606, 0.19458783356196771, 0.75866758737974549]
    # head_mesh = read_off("meshes/head_half_tmc.off")
    # center = np.array(head_mesh[1]).mean(axis=0)
    # grid = None
    # name = "head"
    # compositions.append(
    #     Composition(head_mesh, grid, cam, f"images/{name}.png", center=center))
    #
    # # cam = to_viewpoint(np.array([-0.58538902185339892, -0.18514277826326811, 0.78932999737149179]) * 1)
    # cam = np.array([-10424.616, -15636.341, 11399.555])
    # name = "head_zoom"
    # compositions.append(
    #     Composition(head_mesh, grid, cam, f"images/{name}.png"))

    cam = to_viewpoint(np.array([-0.12036444661000663, 0.49255626199288499, 0.86191689202838073]) * 1)
    grid = to_grid(example_singular_nice())
    name = "singular_nice"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    # cam = to_viewpoint(np.array([0.57252359160177135, 0.72711269019218117, -0.3788454471428917]) * 1.1)
    # grid = to_grid(example_MC_4_without_tunnel())
    cam = to_viewpoint([0.22198612188449593, 0.55432444880051679, 0.80215121214935847])
    grid = to_grid(example_MC_13_with_tunnel())
    name = "blue_intersections"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    cam = to_viewpoint(np.array([0.66712412328021498, 0.31998859433048826, 0.67272037551720032]) * 1.2)
    grid = grid2_singular_manifold()
    name = "singular_manifold"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True))

    ##### results comparison #####

    grid = to_grid(example_single_singular())
    cam = to_viewpoint([0.93177546240610942, 0.28786006350590476, 0.22120368779985883])
    name = "single_singular"
    compositions += render_comparison(grid, cam, name)

    grid = to_grid(example_double_singular())
    cam = to_viewpoint([-0.53190641403250949, 0.32344360546302975, 0.78259810937425145])
    name = "double_singular"
    compositions += render_comparison(grid, cam, name)

    grid = to_grid(example_singular_tunnel())
    cam = to_viewpoint([0.55206138887542744, 0.44226402124762992, 0.70684563974237702])
    name = "singular_tunnel"
    compositions += render_comparison(grid, cam, name)

    grid = to_grid(example_singular_tunnel_2())
    cam = to_viewpoint([-0.86153239068929732, 0.38733023405139694, 0.32823349856904316])
    name = "double_singular_tunnel"
    compositions += render_comparison(grid, cam, name)

    grid = to_grid(example_triple_singular())
    cam = to_viewpoint([0.79777469421428859, 0.42920460883712563, 0.42348428663206422])
    name = "triple_singular"
    compositions += render_comparison(grid, cam, name)

    ##### 2D visualizations #####
    cam = to_viewpoint([0.00099900000199650138, 0.00099900000199650138, 0.999999001998498])

    grid = to_grid(Visualizations.hyperbola())
    name = "face_hyperbola"
    compositions.append(
        Composition(([], []), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=True,
                    show_edges=False, lines=Lines(face_intersection(grid.cell((0, 0, 0))))))

    grid = to_grid(Visualizations.singular())
    name = "face_singular"
    compositions.append(
        Composition(([], []), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=False,
                    show_edges=False, lines=Lines(face_intersection(grid.cell((0, 0, 0))))))

    grid = to_grid(Visualizations.linear())
    name = "face_linear"
    compositions.append(
        Composition(([], []), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=False,
                    show_edges=False, lines=Lines(face_intersection(grid.cell((0, 0, 0))))))

    grid = to_grid(Visualizations.plane())
    name = "face_plane"
    compositions.append(
        Composition(mc_grosso(grid), grid, cam, f"images/{name}.png", show_cubes=True, show_blue_lines=False,
                    show_edges=False))

    grid = to_grid(Visualizations.ambiguous_face_separated())
    name = "ambiguous_face_separated"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_edges=False, lines=Lines(face_intersection(grid.cell((0, 0, 0))))))

    grid = to_grid(Visualizations.ambiguous_face_connected())
    name = "ambiguous_face_connected"
    compositions.append(
        Composition(reference_trilinear(grid), grid, cam, f"images/{name}.png", show_cubes=True,
                    show_blue_lines=False, show_edges=False, lines=Lines(face_intersection(grid.cell((0, 0, 0))))))

    # TODO: show halfedges with direction
    grid = to_grid(Visualizations.hyperbola())
    name = "orientation_hyperbola"
    compositions.append(
        Composition(([], []), grid, cam, f"images/{name}.png", show_cubes=True, show_asymptotes=True))

    grid = to_grid(Visualizations.hyperbola())
    name = "orientation_singular"
    compositions.append(
        Composition(([], []), grid, cam, f"images/{name}.png", show_cubes=True, show_asymptotes=False))

    for comp in compositions:
        vertices, faces = comp.mesh
        renderer.run(blocking=False)
        renderer.clear()
        renderer.update()
        renderer.run(blocking=False)
        if comp.show_mesh:
            renderer.render_mesh(vertices, faces, show_edges=comp.show_edges, duplicate_vertices=comp.duplicate_vertices)

        if comp.lines is not None:
            for i, line in enumerate(comp.lines):
                renderer.render_lines(f"lines{i}", line.points, width=line.width, color=line.color, cylinders=line.cylinders)

        if comp.show_cubes:
            for cube in comp.grid.cells():
                renderer.render_cube(cube)

        if comp.show_blue_lines:
            for cube in comp.grid.cells():
                renderer.render_blue_lines(cube)

        if comp.show_hexagon:
            for cube in comp.grid.cells():
                renderer.render_hexagon(cube)

        if comp.show_asymptotes:
            for cube in comp.grid.cells():
                renderer.render_lines("asymptotes", asymptotes(cube), width=2, color="gray", cylinders=False)

        if comp.show_corners:
            for cube in comp.grid.cells():
                renderer.render_corners(cube)

        # renderer.show()
        renderer.run(blocking=False)
        renderer.set_camera(comp.camera, comp.center)
        renderer.run(blocking=False)
        renderer.update()
        renderer.run(blocking=False)
        renderer.save_to_file(comp.file)
    # renderer.run(blocking=True)

if __name__ == '__main__':
    main()
