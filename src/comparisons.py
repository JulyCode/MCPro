from grid import *
import tables

def mc_skimage(grid: Grid, iso: float = 0):
    values = np.empty(grid.size)
    for idx in grid.indices():
        values[idx[0], idx[1], idx[2]] = grid[idx]
    spacing = np.array(grid.cell_size)
    vertices, faces, _, _ = measure.marching_cubes(values, iso, spacing=spacing)#, method="lorensen"
    return vertices, faces


def mc_grosso(grid: Grid, iso: float = 0):
    vertices_total = []
    vertex_map = {}
    faces_total = []
    for idx in grid.indices():
        if not (idx[0] < grid.size[0] - 1 and idx[1] < grid.size[1] - 1 and idx[2] < grid.size[2] - 1):
            continue
        cube = grid.cell(idx)

        i_case = 0
        for i, v in enumerate(cube.values):
            if v >= iso:
                i_case |= (1 << i)

        vertices = [0] * 12
        for edge in range(12):
            if tables.edge_pattern[i_case] & (1 << edge):
                v0 = edge_vertices[edge][0]
                v1 = edge_vertices[edge][1]

                v0_idx = idx[0] + v0 % 2, idx[1] + (v0 // 2) % 2, idx[2] + (v0 // 4) % 2
                v1_idx = idx[0] + v1 % 2, idx[1] + (v1 // 2) % 2, idx[2] + (v1 // 4) % 2
                vtx = vertex_map.get(grid.edge_id(v0_idx, v1_idx))

                if vtx is not None:
                    vertices[edge] = vtx
                else:
                    interpolation = (iso - cube.values[v0]) / (cube.values[v1] - cube.values[v0])
                    vertices_total.append(cube.local_to_global(cube.edge_to_local(interpolation, edge)))
                    vertices[edge] = len(vertices_total) - 1
                    vertex_map[grid.edge_id(v0_idx, v1_idx)] = vertices[edge]

        for t in range(0, 16, 3):
            t_index = i_case * 16 + t
            if tables.triangle_pattern[t_index] == -1:
                break

            edge0 = tables.triangle_pattern[t_index + 0]
            edge1 = tables.triangle_pattern[t_index + 1]
            edge2 = tables.triangle_pattern[t_index + 2]
            faces_total.append([vertices[edge0], vertices[edge1], vertices[edge2]])

    return vertices_total, faces_total
