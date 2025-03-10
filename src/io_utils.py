import struct

import numpy as np

from grid import Grid


def write_off(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for p in vertices:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for face in faces:
            f.write(f"{len(face)} ")
            for v in face:
                f.write(f"{v} ")
            f.write("\n")


def read_off(filename):
    with open(filename, 'r') as f:
        vertices = []
        faces = []

        f.readline()  # OFF
        n_vertices, n_faces = list(map(int, f.readline().split()))[:2]

        for i in range(n_vertices):
            p = list(map(float, f.readline().split()))
            vertices.append(p)

        for i in range(n_faces):
            face_entry = list(map(int, f.readline().split()))

            if face_entry[0] != 3:
                print("no triangle mesh")
                raise Exception

            face = face_entry[1:]
            faces.append(face)

        return vertices, faces

def read_txt_volume(filename):
    with open(filename) as f:
        size = [int(n) for n in f.readline().split()]
        spacing = [float(x) for x in f.readline().split()]

        grid = Grid(size, spacing)

        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    grid[x, y, z] = float(f.readline())
        # for x, y, z in grid.indices():
        #     grid[x, y, z] = float(f.readline())
        return grid

def read_iso_volume(filename):
    with open(filename, 'rb') as file:
        # Read the three dimensions
        # (nx, ny, nz)
        dimensions = struct.unpack('3i', file.read(3 * 4))  # 3 integers (32 bits each)

        # Read the bounding box
        # (xmin, xmax, ymin, ymax, zmin, zmax)
        bounding_box = struct.unpack('6f', file.read(6 * 4))  # 6 floats (32 bits each)

        # Calculate total number of data points
        total_points = dimensions[0] * dimensions[1] * dimensions[2]

        # Read the volume data (32-bit floats)
        volume_data = struct.unpack(f'{total_points}f', file.read(total_points * 4))  # Total data points as floats

        grid = Grid(dimensions, ((bounding_box[1] - bounding_box[0]) / dimensions[0], (bounding_box[3] - bounding_box[2]) / dimensions[1], (bounding_box[5] - bounding_box[4]) / dimensions[2]))

        nx, ny, nz = dimensions
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    grid[x, y, z] = volume_data[x + y * nx + z * nx * ny]

        return grid
