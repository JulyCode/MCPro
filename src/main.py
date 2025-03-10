import datetime

from tests import *
from comparisons import *

mesh_idx = 0
case_idx = 0

def main():
    renderer = RendererO3D(legacy=True)

    def show_mesh():
        cases = [example_MC_4_with_tunnel(), example_MC_4_without_tunnel(), example_single_singular(), example_singular_nice(), example_he_fail(), example_he_fail(), example_triple_singular(), example_intersecting_planes(), example_complete_hexagon_disc(), example_double_singular(), example_triple_singular(), example_iso_wedge_singular(), example_single_singular(), example_iso_edge(), example_single_singular(), example_cross(), example_complete_hexagon_disc(), example_edge_tunnel(), example_singular_tunnel(), example_singular_tunnel_2(), example_singular_tunnel_2_rotated(), example_intersecting_planes(), example_double_singular(), example_singular_tunnel_open(), example_singular_with_saddle()]

        # grid_random_large_0().cell((0, 3, 5)).values,

        global case_idx
        if case_idx >= len(cases):
            case_idx = 0
        if case_idx < 0:
            case_idx = len(cases) - 1

        grid = Grid((2, 2, 2), (1, 1, 1))
        grid.values = cases[case_idx]

        reference = lambda grid: grid.cell((0, 0, 0)).trilinear_isosurface_mc(subdivision=101)
        meshes = [reference, tmc, mc_skimage, mc_grosso]

        global mesh_idx
        if mesh_idx >= len(meshes):
            mesh_idx = 0
        if mesh_idx < 0:
            mesh_idx = len(meshes) - 1
        vertices, faces = meshes[mesh_idx](grid)

        renderer.clear()
        renderer.render_mesh(vertices, faces, show_edges=(mesh_idx != 0), duplicate_vertices=(mesh_idx != 0))
        cube = grid.cell((0, 0, 0))
        renderer.render_cube(cube)
        renderer.render_blue_lines(cube)
        renderer.render_corners(cube)
        # renderer.render_hexagon(cube)

    def inc_mesh(vis, action, mods):
        if action == 0:
            global mesh_idx
            mesh_idx += 1
            show_mesh()

    def dec_mesh(vis, action, mods):
        if action == 0:
            global mesh_idx
            mesh_idx -= 1
            show_mesh()

    def inc_case(vis, action, mods):
        if action == 0:
            global case_idx
            case_idx += 1
            show_mesh()

    def dec_case(vis, action, mods):
        if action == 0:
            global case_idx
            case_idx -= 1
            show_mesh()

    def save(vis, action, mods):
        if action == 0:
            renderer.save_to_file(f"images/screenshot_{datetime.datetime.now()}.png")

    def cam_pose(vis, action, mods):
        if action == 0:
            print(renderer.get_camera())

    show_mesh()
    renderer.key_callbacks([("left", 263, dec_mesh), ("right", 262, inc_mesh), ("up", 265, dec_case), ("down", 264, inc_case), ("save", 80, save)])#, ("cam", 67, cam_pose)
    renderer.show()
    renderer.run()


if __name__ == '__main__':
    main()
