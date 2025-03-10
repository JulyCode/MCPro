import sys

import sympy
# import pymesh2
import datetime

from cube import *
from io_utils import read_off
from renderer import *


mesh_idx = 1
case_idx = 0

def main():
    renderer = RendererO3D(resolution=(3840*2, 2160*2), legacy=False, transparent_bg=True)

    cases = ["Volumes_half/mecanix_ushort_512_512_743"]
    # "Volumes_half/Carp_ushort_256_256_512"
    # "Volumes_half/mecanix_ushort_512_512_743"
    # "Volumes_half/Porsche_ushort_559_1023_347"
    # "head_half"

    def show_mesh():
        global case_idx
        if case_idx >= len(cases):
            case_idx = 0
        if case_idx < 0:
            case_idx = len(cases) - 1

        meshes = ["trilinear", "tmc", "mc33", "mc"]

        global mesh_idx
        if mesh_idx >= len(meshes):
            mesh_idx = 0
        if mesh_idx < 0:
            mesh_idx = len(meshes) - 1

        # filename = f"meshes/{cases[case_idx]}_{meshes[mesh_idx]}.off"
        filename = f"meshes/{cases[case_idx]}.off"
        vertices, faces = read_off(filename)
        center = np.array(vertices).mean(axis=0)
        # renderer.set_camera(renderer.get_camera(), center=center)

        renderer.clear()
        renderer.render_mesh(vertices, faces, show_edges=False, duplicate_vertices=True)#(mesh_idx != 0)
        # for cube in grid.cells():
        #     renderer.render_cube(cube)
        #     renderer.render_blue_lines(cube)

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
    renderer.key_callbacks(
        [("left", 263, dec_mesh), ("right", 262, inc_mesh), ("up", 265, dec_case), ("down", 264, inc_case),
         ("save", 80, save), ("cam", 67, cam_pose)])
    renderer.show()
    renderer.run()


if __name__ == '__main__':
    main()
