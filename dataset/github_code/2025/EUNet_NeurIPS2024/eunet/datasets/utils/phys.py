import einops
import torch
import numpy as np
import math


def deformation_gradient(triangles, Dm_inv):
    '''
        triangles: bs*n_faces, 3(verts), 3(dim)
        Dm_inv: bs*n_faces, 2, 2
        return: bs*n_faces, 3, 2
    '''
    Ds = get_shape_matrix(triangles)

    return Ds @ Dm_inv


def green_strain_tensor(F):
    '''
        F: bs*n_faces, 3, 2
        return: bs*n_faces, 2, 2
    '''
    device = F.device
    I = torch.eye(2, dtype=F.dtype).to(device)
    Ft = F.permute(0, 2, 1)
    return 0.5 * (Ft @ F - I)


def make_Dm_inv(template_verts, faces): # v is template
    """
    Conpute inverse of the deformation gradient matrix (used in stretching energy loss)
    :param v: vertex positions [Vx3]
    :param f: faces [3xF]
    :return: inverse of the deformation gradient matrix [Fx3x3]
    """
    tri_m = gather_triangles(template_verts.unsqueeze(0), faces)[0] # bs, n_faces, 3(verts), 3(dim)

    edges = get_shape_matrix(tri_m)
    edges = edges.permute(0, 2, 1)
    edges_2d = edges_3d_to_2d(edges).permute(0, 2, 1)
    Dm_inv = torch.inverse(edges_2d)
    return Dm_inv

def edges_3d_to_2d(edges):
    """
    :param edges: Edges in 3D space (in the world coordinate basis) (E, 2, 3)
    :return: Edges in 2D space (in the intrinsic orthonormal basis) (E, 2, 2)
    """
    # Decompose for readability
    device = edges.device

    edges0 = edges[:, 0]
    edges1 = edges[:, 1]

    # Get orthonormal basis
    basis2d_0 = (edges0 / torch.norm(edges0, dim=-1).unsqueeze(-1))
    n = torch.cross(basis2d_0, edges1, dim=-1)
    basis2d_1 = torch.cross(n, edges0, dim=-1)
    basis2d_1 = basis2d_1 / torch.norm(basis2d_1, dim=-1).unsqueeze(-1)

    # Project original edges into orthonormal basis
    edges2d = torch.zeros((edges.shape[0], edges.shape[1], 2)).to(device=device)
    edges2d[:, 0, 0] = (edges0 * basis2d_0).sum(-1)
    edges2d[:, 0, 1] = (edges0 * basis2d_1).sum(-1)
    edges2d[:, 1, 0] = (edges1 * basis2d_0).sum(-1)
    edges2d[:, 1, 1] = (edges1 * basis2d_1).sum(-1)

    return edges2d

def gather_triangles(vertices, faces):
    """
    Generate a tensor of triangles from a tensor of vertices and faces

    :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
    :param faces: LongTensor of shape (num_faces, 3)
    :return: triangles: FloatTensor of shape (batch_size, num_faces, 3, 3)
    """
    F = faces.shape[-1]
    B, V, C = vertices.shape

    vertices = einops.repeat(vertices, 'b m n -> b m k n', k=F)
    faces = einops.repeat(faces, 'm n -> b m n k', k=C, b=B)
    triangles = torch.gather(vertices, 1, faces)

    return triangles


def get_shape_matrix(x):
    '''
        x: bs*n_faces, 3, 3 | bs, n_faces, 3, 3
        return: bs*n_faces, 3(dim), 2(edge)
    '''
    if len(x.shape) == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif len(x.shape) == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def get_face_areas(vertices, faces):
    """
    Computes the area of each face in the mesh

    :param vertices: FloatTensor or numpy array of shape (num_vertices, 3)
    :param faces: LongTensor or numpy array of shape (num_faces, 3)
    :return: areas: FloatTensor or numpy array of shape (num_faces,)
    """
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()

    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = np.abs(np.cross(u, v)) / 2.0
    else:
        out = np.linalg.norm(np.cross(u, v), axis=-1) / 2.0
    return out

def get_face_areas_tensor(vertices, faces):
    """
    Computes the area of each face in the mesh

    :param vertices: FloatTensor or numpy array of shape (num_vertices, 3)
    :param faces: LongTensor or numpy array of shape (num_faces, 3)
    :return: areas: FloatTensor or numpy array of shape (num_faces,)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = torch.abs(torch.cross(u, v)) / 2.0
    else:
        out = torch.linalg.norm(torch.cross(u, v), dim=-1) / 2.0
    return out

def make_pervertex_tensor_from_lens(lens, val_tensor):
    '''
        lens: bs, n_faces
        return: bs*n_faces, 1
    '''
    val_list = []
    for i, n in enumerate(lens):
        val_list.append(val_tensor[i].repeat(n).unsqueeze(-1))
    val_stack = torch.cat(val_list)
    return val_stack

def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:, 0], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 1], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 2], triangle_masses / 3)

    return vertex_masses

def get_vertex_connectivity(faces):
    '''
    edge with face order
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    '''
    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    edges = list(edges)
    return edges

def get_face_connectivity_combined(faces, padding=False):
    assert padding
    """
    Finds the faces that are connected in a mesh
    :param faces: LongTensor of shape (num_faces, 3)
    :return: adjacent_faces: pairs of face indices LongTensor of shape (num_edges, 2)
    :return: adjacent_face_edges: pairs of node indices that comprise the edges connecting the corresponding faces
     LongTensor of shape (num_edges, 2)
    """

    edges = get_vertex_connectivity(faces)

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    adjacent_face_edges = []
    face2edges = [[] for _ in range(len(faces))]

    for key in G:
        if len(G[key]) >= 3:
            G[key] = G[key][:2]
        if len(G[key]) == 1 and padding:
            G[key].append(G[key][0])
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
            adjacent_face_edges += [list(key)]
            cur_edge_idx = len(adjacent_face_edges) - 1
            for f in G[key]:
                if cur_edge_idx not in face2edges[f]:
                    face2edges[f].append(cur_edge_idx)


    adjacent_faces = np.array(adjacent_faces, dtype=np.int64)
    adjacent_face_edges = np.array(adjacent_face_edges, dtype=np.int64)
    face2edges = np.array(face2edges, dtype=np.int64)

    return adjacent_faces, adjacent_face_edges, face2edges