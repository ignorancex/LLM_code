import torch
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import sys
from collections import defaultdict


def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)

def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	E = list(E)
	return E

def direct_edge_in_face(e_id, cur_face):
		e_s = e_id[0]
		s_order = np.where(cur_face == e_s)[0]
		if len(s_order) <= 0:
			return False
		s_order = s_order[0]
		next_order = (s_order+1)%cur_face.shape[0]
		return cur_face[next_order] == e_id[1]

def faces2edges_neighbor_directed(F, vert_mask=None):
	'''
		Output order: receiver -> sender, dst -> src
		# Output is undirected edges (one edge only one direction)
		Output is now BI-DIRECTED edges with local neighbors
	'''
	E = defaultdict(set)
	for f_idx, f in enumerate(F):
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			e_id = tuple([f[i], f[j]])
			e_inv_id = tuple([f[j], f[i]])
			if vert_mask is None:
				E[e_id].add(f_idx)
				E[e_inv_id].add(f_idx)
			else:
				if vert_mask[e_id[1]] == 1:
					# not pinned
					E[e_id].add(f_idx)
				if vert_mask[e_inv_id[1]] == 1:
					E[e_inv_id].add(f_idx)

			
	invert_E = defaultdict(list)
	for key, val in E.items():
		invert_E[len(val)].append(key)
	assert max(invert_E.keys()) == 2

	total_E = invert_E[2] + invert_E[1]
	pair_rst = list()
	e_con_dual_row, e_con_dual_col = [], []
	for e_order, e_id in enumerate(invert_E[2]):
		n_f = list(E[e_id])
		# Find cross id
		neighbors = set()
		for f_idx in n_f:
			for v in F[f_idx]:
				neighbors.add(v)
		assert len(neighbors) == 4
		cross_e = []
		for i in neighbors:
			if i not in e_id:
				cross_e.append(i)
		assert len(cross_e) == 2

		# Check direction
		c_f_idx = 0
		cur_face = F[n_f[c_f_idx]]
		if not direct_edge_in_face(e_id, cur_face):
			c_f_idx += 1
			cur_face = F[n_f[c_f_idx]]
		if not direct_edge_in_face(e_id, cur_face):
			assert False, "Warning about something error of the topo"
			continue
		if cross_e[0] not in cur_face:
			cross_e = [cross_e[1], cross_e[0]]
		assert direct_edge_in_face([cross_e[0], e_id[0]], cur_face)

		# Get neighbor edges
		neighbor_edge = [[cross_e[0], e_id[0]], [e_id[0], cross_e[1]], [cross_e[1], e_id[1]], [e_id[1], cross_e[0]]]
		if vert_mask is not None:
			candidate_neighbor = []
			for n_e in neighbor_edge:
				if vert_mask[n_e[1]] == 1:
					candidate_neighbor.append(n_e)
			neighbor_edge = candidate_neighbor

		# Set edge neighbor connections
		e_con_dual_row.extend([e_order]*len(neighbor_edge))
		for neighbor_e in neighbor_edge:
			assert direct_edge_in_face(neighbor_e, cur_face) or direct_edge_in_face(neighbor_e, F[n_f[(c_f_idx+1)%2]])
			ne_idx = total_E.index(tuple(neighbor_e))
			e_con_dual_col.append(ne_idx)

		# Set face edge
		ce_id = tuple(cross_e)
		pair_rst.append([[e_id[1], e_id[0]], [ce_id[1], ce_id[0]]])

	single_rst = []
	e_con_sgl_row, e_con_sgl_col = [], []
	# Find single edge neighbors
	for e_order, e_id in enumerate(invert_E[1]):
		single_rst.append(list(e_id))
		# Find the rest vert
		n_f = list(E[e_id])
		assert len(n_f) == 1
		cur_face = F[n_f[0]]
		rest_v = cur_face[0]
		for v in cur_face:
			if v not in e_id:
				rest_v = v
				break
		assert v not in e_id
		# Find the neighbor edges
		neighbor_edge = [[e_id[1], rest_v], [rest_v, e_id[0]]]
		if vert_mask is not None:
			candidate_neighbor = []
			for n_e in neighbor_edge:
				if vert_mask[n_e[1]] == 1:
					candidate_neighbor.append(n_e)
			neighbor_edge = candidate_neighbor
		# Find the neighbor edge idx
		e_con_sgl_row.extend([e_order+len(invert_E[2])]*len(neighbor_edge))
		for neighbor_e in neighbor_edge:
			assert not direct_edge_in_face(e_id, cur_face) or direct_edge_in_face(neighbor_e, cur_face)
			ne_idx = total_E.index(tuple(neighbor_e))
			e_con_sgl_col.append(ne_idx)

	e_con_dual = np.stack([e_con_dual_row, e_con_dual_col], axis=0)
	e_con_sgl = np.stack([e_con_sgl_row, e_con_sgl_col], axis=0)
	return pair_rst, single_rst, e_con_dual, e_con_sgl


def edges2graph(E):
	G = {}
	for e in E:
		if not e[0] in G: G[e[0]] = {}
		if not e[1] in G: G[e[1]] = {}
		G[e[0]][e[1]] = 1
		G[e[1]][e[0]] = 1
	return G

def laplacianDict(F, vert_mask=None, with_diag=False, with_neighbor=True):
	'''
		Save/Output Order: receiver, sender (dst, src)
		Note: Target DGL Order: src, dst. Need to care the order when building graph
	'''
	# The outputed E_list are all directed
	if with_neighbor:
		E_pair_list, E_single_list, e_con_dual, e_con_sgl = faces2edges_neighbor_directed(F, vert_mask=vert_mask)
		orig_E = [el[0] for el in E_pair_list] + E_single_list
		G = edges2graph(orig_E)
		row, col, data = [], [], []
		ce_row, ce_col = [], []
		for e_id, ce_id in E_pair_list:
			# Original direction
			n_receiver = len(G[e_id[0]])
			row.append(e_id[0])
			col.append(e_id[1])
			data.append(1.0/n_receiver)
			
			# ce info
			ce_row.append(ce_id[0])
			ce_col.append(ce_id[1])

		for e_id in E_single_list:
			# Original direction
			n_receiver = len(G[e_id[0]])
			row.append(e_id[0])
			col.append(e_id[1])
			data.append(1.0/n_receiver)
	else:
		# The output is undirected
		orig_E = faces2edges(F)
		# Make it directed
		reverse_E = [(i[1], i[0]) for i in orig_E]
		orig_E += reverse_E
		G = edges2graph(orig_E)
		row, col, data = [], [], []
		for e_id in orig_E:
			# Original direction
			n_receiver = len(G[e_id[0]])
			row.append(e_id[0])
			col.append(e_id[1])
			data.append(1.0/n_receiver)

	g_size = [len(G)]*2
	if vert_mask is not None:
		g_size = [vert_mask.shape[0]]*2
	lap_dict = dict(
		indices=np.stack([row, col], axis=0),
		values=data,
		size=g_size)
	if with_neighbor:
		cross_dict = dict(indices=np.stack([ce_row, ce_col], axis=0),
			size=g_size)
		# edge neighbors
		## Offset already moved before merge
		e_con_dict = dict(indices=np.concatenate([e_con_dual, e_con_sgl], axis=-1),
			size=[len(row)+len(ce_row)]*2)
	else:
		# Temporarily, no use in practice
		cross_dict, e_con_dict = lap_dict, lap_dict
	return lap_dict, cross_dict, e_con_dict

def rotateByQuat(p, quat):
	R = np.zeros((3, 3))
	a, b, c, d = quat[3], quat[0], quat[1], quat[2]
	R[0, 0] = a**2 + b**2 - c**2 - d**2
	R[0, 1] = 2 * b * c - 2 * a * d
	R[0, 2] = 2 * b * d + 2 * a * c
	R[1, 0] = 2 * b * c + 2 * a * d
	R[1, 1] = a**2 - b**2 + c**2 - d**2
	R[1, 2] = 2 * c * d - 2 * a * b
	R[2, 0] = 2 * b * d - 2 * a * c
	R[2, 1] = 2 * c * d + 2 * a * b
	R[2, 2] = a**2 - b**2 - c**2 + d**2

	return np.dot(R, p)

def quatFromAxisAngle(axis, angle):
	axis /= np.linalg.norm(axis)

	half = angle * 0.5
	w = np.cos(half)

	sin_theta_over_two = np.sin(half)
	axis *= sin_theta_over_two

	quat = np.array([axis[0], axis[1], axis[2], w])

	return quat


def quatFromAxisAngle_var(axis, angle):
	axis /= torch.norm(axis)

	half = angle * 0.5
	w = torch.cos(half)

	sin_theta_over_two = torch.sin(half)
	axis *= sin_theta_over_two

	quat = torch.cat([axis, w])
	# print("quat size", quat.size())

	return quat

def floyd_map(mesh_faces):
	num_verts = np.max(mesh_faces) + 1
	graph_edges = np.array(faces2edges(mesh_faces))
	receiver = graph_edges[:, 0]
	sender = graph_edges[:, 1]
	cur_graph = sparse.csr_matrix(
		(np.ones(receiver.shape[0]*2), (np.concatenate([receiver, sender], axis=-1), np.concatenate([sender, receiver], axis=-1))), shape=[num_verts, num_verts])
	dist_mat = sparse.csgraph.floyd_warshall(cur_graph, directed=False, unweighted=True)
	return dist_mat

import queue
def get_unneighbor_edges(floyd_dist, center, min_dist=3, first_neighbor=False):
	"""
	Construct several levels of coarse edges

	:param faces: [Fx3] numpy array of faces
	:param center: index od a center node in the mesh (see center of a graph)
	:param n_levels: number of long-range levels to construct
	:return: dictionary of long-range edges for each level
	"""
	node_queue = queue.Queue()
	selected_nodes = set()

	node_mask = list()
	node_queue.put(center)
	while not node_queue.empty():
		cur_node = node_queue.get()
		if cur_node in selected_nodes:
			continue

		node_mask.append(cur_node)
		selected_nodes.add(cur_node)
		# Mask out all with distance less than 4
		neighbors = floyd_dist[cur_node]
		invalid_neighbor = np.where(neighbors <= min_dist)[0]
		for invalid_n in invalid_neighbor:
			selected_nodes.add(invalid_n)
		# To get tight layout
		valid_neighbor = np.where(neighbors == min_dist+1)[0]
		if len(valid_neighbor) <= 0:
			valid_neighbor = np.where(neighbors > min_dist)[0]
		random_selection = np.random.choice(valid_neighbor, size=valid_neighbor.shape[0], replace=False)
		for valid_n in random_selection:
			if valid_n not in selected_nodes:
				node_queue.put(valid_n)
	
	first_neighbor_list = []
	if first_neighbor:
		for cur_node in node_mask:
			neighbors = floyd_dist[cur_node]
			valid_neighbor = np.where(neighbors==1)[0]
			assert len(valid_neighbor) > 0
			for valid_n in valid_neighbor:
				assert valid_n not in first_neighbor_list
				first_neighbor_list.append(valid_n)

	return node_mask, first_neighbor_list
