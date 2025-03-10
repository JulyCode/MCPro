import sys

import sympy
# import pymesh2
import datetime

from cube import *
from io_utils import read_txt_volume, write_off
from renderer import *
from mc import *
from comparisons import *
from tests import *


def reference_trilinear(grid, iso=0):
    total_vertices = []
    total_faces = []
    for cell in grid.cells():
        offset = len(total_vertices)
        vertices, faces = cell.trilinear_isosurface_mc(subdivision=8, iso=iso)
        for v in vertices:
            total_vertices.append(cell.local_to_global(v))
        for v0, v1, v2 in faces:
            total_faces.append((v0 + offset, v1 + offset, v2 + offset))
    return total_vertices, total_faces


mesh_idx = 0
case_idx = 0

def main():
    grid = Grid((2, 2, 2), (1, 1, 1))

    renderer = RendererO3D()

    cases = [grid2_singular_manifold(), grid2_non_manifold_singular(),read_txt_volume("data/skull.txt"), grid2_singular_with_saddle(), grid_extrapolate((32, 32, 32), example_w_saddle_fail_2()),read_txt_volume("data/full_head_half_res.txt"),grid_extrapolate((8, 8, 8), example_w_saddle_fail_1()), grid2_simple(), grid_random_large_0(), grid_random((20, 20, 20))]
    # read_txt_volume("data/Volumes_half/Angio_ushort_384_512_80.txt")
    # grid_extrapolate((16, 16, 16), example_MC_7_with_tunnel()),
    # read_file("data/skull.txt"),
    # read_file("data/full_head.txt"),
    # grid_random_int((50, 50, 50)) ,
    #  read_txt_volume("data/head_ushort_512_512_641_half_res.txt"),
    print("files read")

    def show_mesh():
        global case_idx
        if case_idx >= len(cases):
            case_idx = 0
        if case_idx < 0:
            case_idx = len(cases) - 1

        grid = cases[case_idx]

        meshes = [reference_trilinear, tmc, mc_skimage, mc_grosso]

        global mesh_idx
        if mesh_idx >= len(meshes):
            mesh_idx = 0
        if mesh_idx < 0:
            mesh_idx = len(meshes) - 1

        print("create mesh")
        vertices, faces = meshes[mesh_idx](grid, iso=0) # 950
        print("write mesh")
        write_off("meshes/mesh.off", vertices, faces)

        print("start render")
        renderer.clear()
        renderer.render_mesh(vertices, faces, show_edges=False)#(mesh_idx != 0)
        for cube in grid.cells():
            renderer.render_cube(cube)
            renderer.render_blue_lines(cube)

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

    def new_random(vis, action, mods):
        if action == 0:
            cases[-1] = grid_random((10, 10, 10))
            show_mesh()

    def save(vis, action, mods):
        if action == 0:
            renderer.save_to_file(f"images/screenshot_{datetime.datetime.now()}.png")

    show_mesh()
    renderer.key_callbacks(
        [("left", 263, dec_mesh), ("right", 262, inc_mesh), ("up", 265, dec_case), ("down", 264, inc_case),
         ("save", 80, save), ("random", 82, new_random)])
    renderer.show()
    renderer.run()


if __name__ == '__main__':
    main()
