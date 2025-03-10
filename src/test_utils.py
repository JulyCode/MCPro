
import open3d as o3d
from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
import numpy as np

def isolated_vertices(vertices, faces):
    used = [False] * len(vertices)
    for f in faces:
        for v in f:
            used[v] = True

    for u in used:
        if not u:
            return True
    return False


def duplicate_vertices(vertices, faces):
    for i, v0 in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            v1 = vertices[j]
            if (v0 == v1).all():
                return True
    return False

def duplicate_faces(vertices, faces):
    face_set = set()
    for v0, v1, v2 in faces:
        sorted_face = tuple(sorted((v0, v1, v2)))
        if sorted_face in face_set:
            return True
        face_set.add(sorted_face)
    return False

def to_o3d_mesh(vertices, faces):
    vertices_t = np.array(vertices)
    faces = np.array(faces)
    faces = faces.reshape(faces.shape[0], 3)

    vertices_t = Vector3dVector(vertices_t)
    triangles = Vector3iVector(faces)
    return o3d.geometry.TriangleMesh(vertices_t, triangles)

def edge_manifold(vertices, faces):
    if len(vertices) == 0:
        return True
    mesh = to_o3d_mesh(vertices, faces)
    return mesh.is_edge_manifold(allow_boundary_edges=True)

def vertex_manifold(vertices, faces):
    if len(vertices) == 0:
        return True
    mesh = to_o3d_mesh(vertices, faces)
    return mesh.is_vertex_manifold()

def self_intersecting(vertices, faces):
    if len(vertices) == 0:
        return False
    mesh = to_o3d_mesh(vertices, faces)
    return mesh.is_self_intersecting()

def orientable(vertices, faces):
    if len(vertices) == 0:
        return False
    mesh = to_o3d_mesh(vertices, faces)
    return mesh.is_orientable()

def connected_components(vertices, faces):
    if len(vertices) == 0:
        return 0
    mesh = to_o3d_mesh(vertices, faces)
    _, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    return len(cluster_n_triangles)

def genus(vertices, faces):
    if len(vertices) == 0:
        return 1
    # return 1 - (count_boundary_components(vertices, faces) + euler_characteristics(vertices, faces)) / 2

    V = len(vertices)

    # Set to collect unique edges
    edges = set()
    for i, f in enumerate(faces):
        v1, v2, v3 = f
        edges.add(tuple(sorted((v1, v2))))
        edges.add(tuple(sorted((v2, v3))))
        edges.add(tuple(sorted((v3, v1))))

    E = len(edges)
    return E - V + connected_components(vertices, faces)

def euler_characteristics(vertices, faces):
    V = len(vertices)
    F = len(faces)

    # Set to collect unique edges
    edges = set()
    for i, f in enumerate(faces):
        v1, v2, v3 = f
        edges.add(tuple(sorted((v1, v2))))
        edges.add(tuple(sorted((v2, v3))))
        edges.add(tuple(sorted((v3, v1))))

    E = len(edges)
    return V - E + F

def euler_characteristics_he(vertices, faces, halfedges, next_halfedges, opposite):
    V = len(vertices)

    contours = []
    visited = set()
    for i, he in enumerate(halfedges):
        if i in visited:
            continue
        contour = [i]
        visited.add(i)
        next_he = next_halfedges.get(i)
        while next_he is not None and next_he != i:
            if next_he in visited:
                # print("loop")
                break
            contour.append(next_he)
            visited.add(next_he)
            next_he = next_halfedges.get(next_he)
        if next_he == i:
            contours.append(contour)

    F = len(contours)

    # Set to collect unique edges
    edges = set()
    for i, _ in enumerate(halfedges):
        opp = opposite.get(i)
        edges.add(tuple(sorted((i, opp if opp is not None else -1))))

    E = len(edges)
    return V - E + F


def count_boundary_components(vertices, faces):
    # Dictionary to keep track of edge counts
    edge_count = {}

    for v1, v2, v3 in faces:
        # Since the faces are defined by vertex indices (triangles), create edges
        edges = [
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v3))),
            tuple(sorted((v3, v1))),
        ]

        for edge in edges:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    # Count boundary edges (those that appear only once)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Now we need to count connected components in boundary edges
    visited = set()
    boundary_components = 0

    def dfs(edge, visited_edges):
        stack = [edge]
        while stack:
            current_edge = stack.pop()
            if current_edge in visited_edges:
                continue
            visited_edges.add(current_edge)
            # Find adjacent edges to perform DFS
            for neighbor in boundary_edges:
                if neighbor not in visited_edges and (current_edge[0] in neighbor or current_edge[1] in neighbor):
                    stack.append(neighbor)

    for edge in boundary_edges:
        if edge not in visited:
            boundary_components += 1  # Found a new boundary component
            dfs(edge, visited)

    return boundary_components

def betti_0(vertices, faces):
    return connected_components(vertices, faces)

def betti_1(vertices, faces):
    # return 2 * genus(vertices, faces)
    return betti_0(vertices, faces) + betti_2(vertices, faces) - euler_characteristics(vertices, faces)

def betti_1_he(vertices, faces, halfedges, next_halfedges, opposite):
    # return 2 * genus(vertices, faces)
    return betti_0(vertices, faces) + betti_2(vertices, faces) - euler_characteristics_he(vertices, faces, halfedges, next_halfedges, opposite)

def betti_2(vertices, faces):
    # return euler_characteristics(vertices, faces) - betti_0(vertices, faces) + betti_1(vertices, faces)
    return connected_components(vertices, faces) # only for closed surfaces
