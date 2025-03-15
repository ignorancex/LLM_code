import torch
import torch.nn.functional as F

from mmcv.ops import grouping_operation
from eunet.datasets.utils.hood_common import gather, unsorted_segment_sum


def face_normals_batched(verts, faces, normalized=True, eps=1e-7, with_face_area=False):
	'''
		bs, num_verts, 3
		bs, num_faces, 3
	'''
	# face_verts: bs, 3(coordinate), num_faces, 3(face)
	face_verts = grouping_operation(verts.transpose(-1, -2), faces.int()).permute(0, 2, 3, 1)
	face_normals = torch.cross(
		face_verts[:, :, 2] - face_verts[:, :, 1],
		face_verts[:, :, 0] - face_verts[:, :, 1],
		dim=-1,
	)

	face_area = torch.norm(face_normals, dim=-1, keepdim=True)
	if normalized:
		face_normals = face_normals / (face_area + eps)
	if with_face_area:
		return face_normals, face_area
	else:
		return face_normals

def vertex_normal_batched(verts, faces, v2f_mask_sparse, normalized=True, eps=1e-7):
	bs, v_num, f_num = v2f_mask_sparse.shape
	face_normals = face_normals_batched(verts, faces, normalized=normalized, eps=eps)
	vert_normals = torch.bmm(v2f_mask_sparse, face_normals)
	assert normalized
	if normalized:
		vert_normals = vert_normals / (torch.linalg.norm(vert_normals, dim=-1, ord=2, keepdim=True) + eps)
	return vert_normals

def vertex_normal_batched_simple(vertices, faces):
	'''
		v: bs, n_verts, 3dim
		f: bs, n_face, 3nodes
		return: bs, n_verts, 3dim
	'''
	v = vertices
	f = faces

	# bs, n_face, 3nodes, 3dim
	triangles = gather(v, f, 1, 2, 2)

	# Compute face normals
	v0, v1, v2 = torch.unbind(triangles, dim=-2)
	e0 = v1 - v0
	e1 = v2 - v1
	e2 = v0 - v2
	face_normals = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)  # F x 3

	vn = unsorted_segment_sum(face_normals, f, 1, 2, 2)

	vn = F.normalize(vn, dim=-1)
	return vn
	