
from cube import *


class Grid:
    def __init__(self, size: (int, int, int), cell_size: Point3):
        self.size = np.array(size, dtype=int)
        self.cell_size = np.array(cell_size)
        self.values = np.zeros(size[0] * size[1] * size[2])

    def __getitem__(self, item: (int, int, int)) -> float:
        return self.values[(item[2] * self.size[1] + item[1]) * self.size[0] + item[0]]

    def __setitem__(self, item: (int, int, int), value: float):
        self.values[(item[2] * self.size[1] + item[1]) * self.size[0] + item[0]] = value

    def position(self, idx: (int, int, int)) -> Point3:
        return idx * self.cell_size

    def cell(self, idx: (int, int, int)) -> Cube:
        min_pos = self.position(idx)
        max_pos = self.position((idx[0] + 1, idx[1] + 1, idx[2] + 1))
        cube = Cube((min_pos, max_pos))

        cube[0, 0, 0] = self[idx]
        cube[0, 0, 1] = self[idx[0], idx[1], idx[2] + 1]
        cube[0, 1, 0] = self[idx[0], idx[1] + 1, idx[2]]
        cube[0, 1, 1] = self[idx[0], idx[1] + 1, idx[2] + 1]
        cube[1, 0, 0] = self[idx[0] + 1, idx[1], idx[2]]
        cube[1, 0, 1] = self[idx[0] + 1, idx[1], idx[2] + 1]
        cube[1, 1, 0] = self[idx[0] + 1, idx[1] + 1, idx[2]]
        cube[1, 1, 1] = self[idx[0] + 1, idx[1] + 1, idx[2] + 1]
        return cube

    def indices(self) -> [(int, int, int)]:
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    yield np.array((i, j, k), dtype=int)

    def cells(self) -> [Cube]:
        for idx in self.indices():
            if idx[0] < self.size[0] - 1 and idx[1] < self.size[1] - 1 and idx[2] < self.size[2] - 1:
                yield self.cell(idx)

    def edges(self) -> ((int, int, int), (int, int, int)):
        for idx in self.indices():
            if idx[0] < self.size[0] - 1:
                yield np.array((idx, (idx[0] + 1, idx[1], idx[2])), dtype=int)
            if idx[1] < self.size[1] - 1:
                yield np.array((idx, (idx[0], idx[1] + 1, idx[2])), dtype=int)
            if idx[2] < self.size[2] - 1:
                yield np.array((idx, (idx[0], idx[1], idx[2] + 1)), dtype=int)

    def faces(self) -> ((int, int, int), (int, int, int), (int, int, int), (int, int, int)):
        for idx in self.indices():
            if idx[0] < self.size[0] - 1 and idx[1] < self.size[1] - 1:
                yield np.array((idx, (idx[0] + 1, idx[1], idx[2]), (idx[0], idx[1] + 1, idx[2]), (idx[0] + 1, idx[1] + 1, idx[2])), dtype=int)
            if idx[0] < self.size[0] - 1 and idx[2] < self.size[2] - 1:
                yield np.array((idx, (idx[0], idx[1], idx[2] + 1), (idx[0] + 1, idx[1], idx[2]), (idx[0] + 1, idx[1], idx[2] + 1)), dtype=int)
            if idx[1] < self.size[1] - 1 and idx[2] < self.size[2] - 1:
                yield np.array((idx, (idx[0], idx[1] + 1, idx[2]), (idx[0], idx[1], idx[2] + 1), (idx[0], idx[1] + 1, idx[2] + 1)), dtype=int)

    def vertex_id(self, v):
        return (v[2] * self.size[1] + v[1]) * self.size[0] + v[0]

    def edge_id(self, v0, v1):
        id0 = self.vertex_id(v0)
        id1 = self.vertex_id(v1)
        return (id0, id1) if id0 < id1 else (id1, id0)

    def face_id(self, v0, v1, v2, v3):
        vertices = (self.vertex_id(v0), self.vertex_id(v1), self.vertex_id(v2), self.vertex_id(v3))
        argsort = sorted(range(4), key=vertices.__getitem__)
        return vertices[argsort[0]], vertices[argsort[1]], vertices[argsort[2]]
