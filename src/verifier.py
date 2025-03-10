import os

from io_utils import read_iso_volume, write_off
from mc import tmc, tmc_halfedges
from paper_images import reference_trilinear
from tests import *
from test_utils import euler_characteristics, betti_0, betti_1, betti_2, euler_characteristics_he, betti_1_he


def read_euler(filename):
    with open(filename) as f:
        return int(f.readline())

def read_betti(filename):
    with open(filename) as f:
        return int(f.readline()), int(f.readline())


def verify_euler():
    entries = os.listdir("data/MarchingCubes_cases/Grid_invariants/")
    num_tests = 10000

    for i in range(num_tests):
        grid = read_iso_volume(f"data/MarchingCubes_cases/Grids/{i}-scalar_field.iso")

        vertices, faces, halfedges, next_he, opposite = tmc_halfedges(grid)
        euler = euler_characteristics_he(vertices, faces, halfedges, next_he, opposite)
        solution = read_euler(f"data/MarchingCubes_cases/Grid_invariants/{i}-euler_number.txt")

        if euler != solution:
            print(f"error in test {i}: euler {euler} != {solution}")
            raise Exception(f"error in test {i}")


def verify_betti():
    entries = os.listdir("data/Closed_Surfaces/InvariantsGrid/")
    num_tests = 10000

    for i in range(num_tests):
        grid = read_iso_volume(f"data/Closed_Surfaces/Grids/{i}-scalar_field.iso")

        vertices, faces, halfedges, next_he, opposite = tmc_halfedges(grid)
        b0 = betti_0(vertices, faces)
        b1 = betti_1_he(vertices, faces, halfedges, next_he, opposite)
        b2 = betti_2(vertices, faces)
        solution = read_betti(f"data/Closed_Surfaces/InvariantsGrid/{i}-invariant_grid.txt")

        if b0 != solution[0]:
            print(f"error in test {i}: b0 {b0} != {solution[0]}")
            raise Exception(f"error in test {i}")
        if b1 != solution[1] * 2:
            print(f"error in test {i}: b1 {b1} != {solution[1] * 2}")
            raise Exception(f"error in test {i}")


def debug_test(single_cell, test_number):
    if single_cell:
        grid = read_iso_volume(f"data/MarchingCubes_cases/Grids/{test_number}-scalar_field.iso")
    else:
        grid = read_iso_volume(f"data/Closed_Surfaces/Grids/{test_number}-scalar_field.iso")

    vertices, faces, halfedges, next_he, opposite = tmc_halfedges(grid)
    euler = euler_characteristics(vertices, faces)
    write_off("meshes/mesh.off", vertices, faces)
    vertices, faces = reference_trilinear(grid, subdivision=51)

    renderer = RendererO3D(legacy=True)
    renderer.clear()
    renderer.render_mesh(vertices, faces, show_edges=False, duplicate_vertices=True)
    for cube in grid.cells():
        renderer.render_cube(cube)
        renderer.render_blue_lines(cube)

    renderer.show()
    renderer.run()


def main():
    # debug_test(True, 0)

    verify_euler()
    verify_betti()
    print("Tests passed")



if __name__ == '__main__':
    main()