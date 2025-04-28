from zac.ds.architecture import Architecture
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import math
from copy import deepcopy

class VertexMatchingPlacer:
    """class to find a qubit layout via SA."""

    def __init__(self, mapping: list, l2: bool = False):
        self.mapping = [mapping]
        self.l2 = l2
        self.print_detail = False
        # self.cost_atom_transfer = pow(0.999, 2)
        self.cost_atom_transfer = 0.9999
        self.n_qubit = len(mapping)

    def run(self, architecture, qubit_mapping, list_gate, dynamic_placement, list_reuse_qubits):
        self.architecture = architecture
        self.list_reuse_qubit = list_reuse_qubits
        print("[INFO] ZAC: Minimum-weight-full-matching-based intermediate placement: Start")
        # print("[INFO] ZAC: Minimum-weight-full-matching-based intermediate placement for layer {}: Start".format(0))
        if self.print_detail:
            print("[INFO]               Gate placement: Start")
        self.place_gate(qubit_mapping, list_gate[0:2], 0, False)
        # print(self.mapping[0])
        # print('Hi')
        for layer in range(len(list_gate)):
            # the case that don't reuse qubits
            if dynamic_placement:
                if self.print_detail:
                    print("[INFO]               Qubit placement: Start")
                self.place_qubit(list_gate[layer:], layer, False)
            else:
                self.mapping.append(deepcopy(self.mapping[0]))  # keep the initial mapping for static placement
            # print(self.mapping[0])
            # print('Hi2')
            if layer + 1 < len(list_gate):
                if self.print_detail:
                    print("[INFO]               Gate placement: Start")
                # print(self.mapping[-2])
                # print(self.mapping[-1])
                self.place_gate(self.mapping[-2:], list_gate[layer+1:layer+3], layer+1, False)
            # the case that reuse qubits
            if len(list_reuse_qubits[layer]) > 0:
                if dynamic_placement:
                    if self.print_detail:
                        print("[INFO]               Qubit placement: Start")
                    self.place_qubit(list_gate[layer:], layer, True)
                else:
                    self.mapping.append(deepcopy(self.mapping[0]))  # keep the initial mapping for static placement
                    for q in list_reuse_qubits[layer]:
                        self.mapping[-1][q] = self.mapping[-4][q] 
                if layer + 1 < len(list_gate):
                    if self.print_detail:
                        print("[INFO]               Gate placement: Start")
                    self.place_gate([self.mapping[-4], self.mapping[-1]], list_gate[layer+1:layer+3], layer+1, True)
                    # keep the mapping with shorter distance
                    self.filter_mapping(layer)

        print("[INFO] ZAC: Minimum-weight-full-matching-based intermediate placement: Finish")

    def filter_mapping(self, layer):
        # cost for mapping without reuse
        last_gate_mapping = self.mapping[-5]
        qubit_mapping = self.mapping[-4]
        gate_mapping = self.mapping[-3]
        cost_no_reuse = 0
        movement_parallel_movement_1 = dict()
        movement_parallel_movement_2 = dict()
        
        # print("non reuse: ")
        for q in range(len(last_gate_mapping)):
            if last_gate_mapping[q] != gate_mapping[q]:
                # print("q: {}, loc1: {}, loc2: {}".format(q, last_gate_mapping[q], qubit_mapping[q]))
                # ! gate mapping slm idx
                slm_idx1 = self.architecture.dict_SLM[last_gate_mapping[q][0]].entanglement_id
                slm_idx2 = self.architecture.dict_SLM[qubit_mapping[q][0]].entanglement_id
                key = (slm_idx1, last_gate_mapping[q][1], slm_idx2, qubit_mapping[q][1])
                dis = self.architecture.distance(last_gate_mapping[q][0], last_gate_mapping[q][1], last_gate_mapping[q][2], qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2])
                if key in movement_parallel_movement_1:
                    movement_parallel_movement_1[key] = max(movement_parallel_movement_1[key], dis)
                else:
                    movement_parallel_movement_1[key] = dis
                # cost_no_reuse += math.sqrt(self.architecture.distance(last_gate_mapping[q][0], last_gate_mapping[q][1], last_gate_mapping[q][2], qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2]))
            if qubit_mapping[q] != gate_mapping[q]:
                # print("q: {}, loc1: {}, loc2: {}".format(q, qubit_mapping[q], gate_mapping[q]))
                slm_idx1 = self.architecture.dict_SLM[gate_mapping[q][0]].entanglement_id
                slm_idx2 = self.architecture.dict_SLM[qubit_mapping[q][0]].entanglement_id
                key = (slm_idx2, qubit_mapping[q][1], slm_idx1, gate_mapping[q][1])
                dis = self.architecture.distance(qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2], gate_mapping[q][0], gate_mapping[q][1], gate_mapping[q][2])
                if key in movement_parallel_movement_2:
                    movement_parallel_movement_2[key] = max(movement_parallel_movement_2[key], dis)
                else:
                    movement_parallel_movement_2[key] = dis
                # cost_no_reuse += math.sqrt(self.architecture.distance(qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2], gate_mapping[q][0], gate_mapping[q][1], gate_mapping[q][2]))
        # print(movement_parallel_movement_1)
        # print(movement_parallel_movement_2)
        for key in movement_parallel_movement_1:
            cost_no_reuse += math.sqrt(movement_parallel_movement_1[key])
        for key in movement_parallel_movement_2:
            cost_no_reuse += math.sqrt(movement_parallel_movement_2[key])
        # cost for mapping with reuse
        gate_mapping = self.mapping[-1]
        qubit_mapping = self.mapping[-2]
        cost_reuse = 0
        movement_parallel_movement_1 = dict()
        movement_parallel_movement_2 = dict()
        # print("reuse: ")
        for q in range(len(last_gate_mapping)):
            if last_gate_mapping[q] != gate_mapping[q]:
                # print("q: {}, loc1: {}, loc2: {}".format(q, last_gate_mapping[q], qubit_mapping[q]))
                slm_idx1 = self.architecture.dict_SLM[last_gate_mapping[q][0]].entanglement_id
                slm_idx2 = self.architecture.dict_SLM[qubit_mapping[q][0]].entanglement_id
                key = (slm_idx1, last_gate_mapping[q][1], slm_idx2, qubit_mapping[q][1])
                dis = self.architecture.distance(last_gate_mapping[q][0], last_gate_mapping[q][1], last_gate_mapping[q][2], qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2])
                if key in movement_parallel_movement_1:
                    movement_parallel_movement_1[key] = max(movement_parallel_movement_1[key], dis)
                else:
                    movement_parallel_movement_1[key] = dis
                # cost_reuse += math.sqrt(self.architecture.distance(last_qubit_mapping[q][0], last_qubit_mapping[q][1], last_qubit_mapping[q][2], gate_mapping[q][0], gate_mapping[q][1], gate_mapping[q][2]))
            if qubit_mapping[q] != gate_mapping[q]:
                # print("q: {}, loc1: {}, loc2: {}".format(q, qubit_mapping[q], gate_mapping[q]))
                slm_idx1 = self.architecture.dict_SLM[gate_mapping[q][0]].entanglement_id
                slm_idx2 = self.architecture.dict_SLM[qubit_mapping[q][0]].entanglement_id
                key = (slm_idx2, qubit_mapping[q][1], slm_idx1, gate_mapping[q][1])
                dis = self.architecture.distance(qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2], gate_mapping[q][0], gate_mapping[q][1], gate_mapping[q][2])
                if key in movement_parallel_movement_2:
                    movement_parallel_movement_2[key] = max(movement_parallel_movement_2[key], dis)
                else:
                    movement_parallel_movement_2[key] = dis
                # cost_reuse += math.sqrt(self.architecture.distance(qubit_mapping[q][0], qubit_mapping[q][1], qubit_mapping[q][2], gate_mapping[q][0], gate_mapping[q][1], gate_mapping[q][2]))
        # print(movement_parallel_movement_1)
        # print(movement_parallel_movement_2)
        for key in movement_parallel_movement_1:
            cost_reuse += math.sqrt(movement_parallel_movement_1[key])
        for key in movement_parallel_movement_2:
            cost_reuse += math.sqrt(movement_parallel_movement_2[key])
        # print("before")
        # for p in self.mapping:
        #     print(p[0:10])
        # print("self.cost_atom_transfer: ", self.cost_atom_transfer)
        # print("cost_no_reuse: ", cost_no_reuse)
        # print("cost_no_reuse: ", self.cost_atom_transfer * pow((1 - cost_no_reuse/1.5e6), self.n_qubit))
        # print("cost_reuse: ", cost_reuse)
        # print("cost_reuse: ", pow((1 - cost_reuse/1.5e6), self.n_qubit))
        if self.cost_atom_transfer * pow((1 - cost_no_reuse/1.5e6), self.n_qubit) >  pow((1 - cost_reuse/1.5e6), self.n_qubit):
        # if False: # !
            # print("no reuse ")
            self.list_reuse_qubit[layer] = []
            self.mapping.pop(-1)
            self.mapping.pop(-1)
        else:
            # print("reuse reuse ")
            self.mapping.pop(-3)
            self.mapping.pop(-3)
        # print("after")
        # for p in self.mapping:
        #     print("===")
        #     print(p[0:10])
        #     print(p[40:50])
        # assert(0)

    def place_gate(self, list_qubit_mapping: list, list_two_gate_layer: list, layer: int, test_reuse: bool):
        """
        generate gate mapping based on minimum weight matching
        qubit_mapping: the initial mapping before the Rydberg stage
        list_gate: gates to be executed in the current Rydberg stage
        """
        # construct the sparse matrix A for scipy to find minimum weight matching
        # use list_row_coo, list_col_coo, and data to represent A where A[list_i[k], list_j[k]] = data[k]
        # ! when having multiple entangling zone, we may not collect enough sites
        list_gate = list_two_gate_layer[0]
        dict_reuse_qubit_neighbor = dict()
        if len(list_two_gate_layer) > 1 and test_reuse:
            for q in self.list_reuse_qubit[layer]:
                for gate in list_two_gate_layer[1]:
                    if q == gate[0]:
                        dict_reuse_qubit_neighbor[q] = gate[1]
                        break
                    elif q == gate[1]:
                        dict_reuse_qubit_neighbor[q] = gate[0]
                        break

        if layer > 0:
            gate_mapping = list_qubit_mapping[0]
            # print("gate mapping: ", gate_mapping)
            qubit_mapping = list_qubit_mapping[1]
        else:
            qubit_mapping = list_qubit_mapping[0]
        site_Rydberg_to_idx = dict()
        list_Rydberg = []
        list_row_coo = []
        list_col_coo = []
        list_data = []
        # store data in coo form
        # row: rydberg site
        # col: gate id
        # sample_slm = self.architecture.dict_SLM[self.architecture.entanglement_zone[0][0]]
        expand_factor = math.ceil(math.sqrt(len(list_gate)) / 2)
        # print("expand_factor = {}".format(expand_factor))
        for i, gate in enumerate(list_gate):
            q1 = gate[0]
            q2 = gate[1]
            set_nearby_site = set()

            if  test_reuse and (q1 in self.list_reuse_qubit[layer - 1]):
                location = gate_mapping[q1]
                slm_idx = self.architecture.entanglement_zone[self.architecture.dict_SLM[location[0]].entanglement_id][0]
                set_nearby_site.add((slm_idx, location[1], location[2]))
            elif test_reuse and (q2 in self.list_reuse_qubit[layer - 1]):
                location = gate_mapping[q2]
                slm_idx = self.architecture.entanglement_zone[self.architecture.dict_SLM[location[0]].entanglement_id][0]
                set_nearby_site.add((slm_idx, location[1], location[2]))
            else:
                # print("Gate {}: {}->{}, {}->{}".format(i, q1, qubit_mapping[q1], q2, qubit_mapping[q2]))
                slm = self.architecture.dict_SLM[qubit_mapping[q1][0]]
                list_nearest_site = self.architecture.nearest_entanglement_site(qubit_mapping[q1][0], qubit_mapping[q1][1], qubit_mapping[q1][2], qubit_mapping[q2][0], qubit_mapping[q2][1], qubit_mapping[q2][2])
                list_nearest_site += self.architecture.nearest_entanglement_site(qubit_mapping[q1][0], 0, qubit_mapping[q1][2], qubit_mapping[q2][0], 0, qubit_mapping[q2][2])
                list_nearest_site += self.architecture.nearest_entanglement_site(qubit_mapping[q1][0], slm.n_r - 1, qubit_mapping[q1][2], qubit_mapping[q2][0], slm.n_r - 1, qubit_mapping[q2][2])
                # print("nearest_site: {}".format(nearest_site))
                # collect nearby sites
                list_set = set(list_nearest_site)
                for nearest_site in list_nearest_site:
                    set_nearby_site.add(nearest_site)
                    slm_idx = nearest_site[0]
                    slm = self.architecture.dict_SLM[slm_idx]
                    slm_r = nearest_site[1]
                    slm_c = nearest_site[2]
                    low_r = max(0, slm_r - expand_factor)
                    high_r = min(slm.n_r, slm_r + expand_factor + 1)
                    low_c = max(0, slm_c - expand_factor)
                    high_c = min(slm.n_c, slm_c + expand_factor + 1)
                    if high_c - low_c < 2 * expand_factor:
                        height_gap = math.ceil(len(list_gate) // (high_c - low_c)) - expand_factor
                        low_r = max(0, low_r - height_gap // 2)
                        high_r = min(slm.n_r, low_r + height_gap + expand_factor)
                    if high_r - low_r < 2 * expand_factor:
                        width_gap = math.ceil(len(list_gate) / (high_r - low_r)) - expand_factor
                        low_c = max(0, low_c - width_gap // 2)
                        high_c = min(slm.n_c, low_c + width_gap + expand_factor)
                    for r in range(low_r, high_r):
                        for c in range(low_c, high_c):
                            set_nearby_site.add((slm_idx, r, c))
                # print(expand_factor)
                # print((low_r, high_r, low_c, high_c))
            # print("set_nearby_site: {}".format(set_nearby_site))
            for site in set_nearby_site:
                if site not in site_Rydberg_to_idx:
                    site_Rydberg_to_idx[site] = len(list_Rydberg)
                    list_Rydberg.append(site)
                idx_rydberg = site_Rydberg_to_idx[site]
                dis1 = self.architecture.distance(qubit_mapping[q1][0], qubit_mapping[q1][1], qubit_mapping[q1][2], site[0], site[1], site[2])
                dis2 = self.architecture.distance(qubit_mapping[q2][0], qubit_mapping[q2][1], qubit_mapping[q2][2], site[0], site[1], site[2])
                dis3 = 0
                q3 = -1
                if q1 in dict_reuse_qubit_neighbor:
                    q3 = dict_reuse_qubit_neighbor[q1]
                elif q2 in dict_reuse_qubit_neighbor:
                    q3 = dict_reuse_qubit_neighbor[q2]
                if q3 > -1:
                    dis3 = self.architecture.distance(qubit_mapping[q3][0], qubit_mapping[q3][1], qubit_mapping[q3][2], site[0], site[1], site[2])
                list_row_coo.append(idx_rydberg)
                list_col_coo.append(i)
                # print("add edge site {} to gate {}".format(idx_rydberg, i))
                if qubit_mapping[q1][1] == qubit_mapping[q2][1] and qubit_mapping[q1][0] == qubit_mapping[q2][0]:
                    list_data.append(math.sqrt(max(dis1, dis2)) + math.sqrt(dis3))
                else:
                    list_data.append(math.sqrt(dis1) + math.sqrt(dis2) + math.sqrt(dis3))
                
        # input()
        # if there is no enough site for all gates
        if len(list_Rydberg) < len(list_gate):
        # if True:
            print(layer)
            print(self.list_reuse_qubit[layer - 1])
            print(list_Rydberg)
            for gate in list_gate:
                print("gate: ", gate)
                for q in gate:
                    print(qubit_mapping[q])
            print("[Error] ZAC: Minimum-weight-full-matching-based intermediate placement: No enough sites for gates ({} vs {}).".format(len(list_Rydberg), len(list_gate)))
            assert(0)
        # print("[Error] ZAC: Minimum-weight-full-matching-based intermediate placement: No enough sites for gates ({} vs {}).".format(len(list_Rydberg), len(list_gate)))
        # print(list_Rydberg)
        # print(list_gate)
        np_data = np.array(list_data)
        np_col_coo = np.array(list_col_coo)
        np_row_coo = np.array(list_row_coo)
        matrix = coo_matrix((np_data, (np_row_coo, np_col_coo)), shape=(len(list_Rydberg), len(list_gate)))
        # solve minimal matching by scipy
        site_ind, gate_ind = min_weight_full_bipartite_matching(matrix)
        cost = matrix.toarray()[site_ind, gate_ind].sum()
        if self.print_detail:
            print("[INFO]               Gate placement cost: {}".format(cost))
        # process the solution
        tmp_mapping = deepcopy(qubit_mapping)
        for idx_rydberg, idx_gate in zip(site_ind, gate_ind):
            q0 = list_gate[idx_gate][0]
            q1 = list_gate[idx_gate][1]
            site = list_Rydberg[idx_rydberg]
            if  test_reuse and (q0 in self.list_reuse_qubit[layer - 1]):
                tmp_mapping[q0] = gate_mapping[q0]
                if site == gate_mapping[q0]:
                    tmp_mapping[q1] = (site[0]+1, site[1], site[2])
                else:
                    tmp_mapping[q1] = site
            elif test_reuse and (q1 in self.list_reuse_qubit[layer - 1]):
                tmp_mapping[q1] = gate_mapping[q1]
                if site == gate_mapping[q1]:
                    tmp_mapping[q0] = (site[0]+1, site[1], site[2])
                else:
                    tmp_mapping[q0] = site
            else:
                # check if q0.c < q1.c, then q0 can be in the left SLM array, and q1 should be in the right SLM array of the Rydberg site
                if qubit_mapping[q0][2] < qubit_mapping[q1][2]:
                    tmp_mapping[q0] = site
                    tmp_mapping[q1] = (site[0]+1, site[1], site[2])
                else:
                    tmp_mapping[q0] = (site[0]+1, site[1], site[2])
                    tmp_mapping[q1] = site
        self.mapping.append(tmp_mapping)
    
    def place_qubit(self, list_gate: list, layer: int, test_reuse: bool):
        """
        generate qubit mapping based on minimum weight matching
        list_gate: list_gate[0]: gates to be executed in the current Rydberg stage. list_gate[1:] is the list of unexecuted gates
        """
        # construct a Boolean array to record the sites occupied by the qubits
        qubit_mapping = self.mapping[0]
        if test_reuse:
            last_gate_mapping = self.mapping[-3]
        else:
            last_gate_mapping = self.mapping[-1]
        is_empty_storage_site = dict()
        qubit_to_place = []
        for slm_id in self.architecture.storage_zone:
            slm = self.architecture.dict_SLM[slm_id]
            is_empty_storage_site[slm_id] = [[True for j in range(slm.n_c)] for i in range(slm.n_r)]
        # print("current mapping before inter qubit placement")
        # print(self.mapping[-1])
        for q, mapping in enumerate(last_gate_mapping):
            array_id = mapping[0]
            if array_id in is_empty_storage_site:
                is_empty_storage_site[array_id][mapping[1]][mapping[2]] = False
            elif (not test_reuse) or (q not in self.list_reuse_qubit[layer]):
                qubit_to_place.append(q)
        # print(qubit_to_place)
        common_site = set()
        for q, mapping in enumerate(self.mapping[0]):
            array_id = mapping[0]
            if array_id in is_empty_storage_site:
                if is_empty_storage_site[array_id][mapping[1]][mapping[2]]:
                    common_site.add((array_id, mapping[1], mapping[2]))
            else:
                is_empty_storage_site[array_id] = [[True for j in range(slm.n_c)] for i in range(slm.n_r)]
                common_site.add((array_id, mapping[1], mapping[2]))
        # print(is_empty_storage_site[0][9][0])
        dict_qubit_interaction = dict()
        for q in qubit_to_place:
            dict_qubit_interaction[q] = []
        

        if len(list_gate) > 1:
            for gate in list_gate[1]:
                if gate[0] in dict_qubit_interaction and ((not test_reuse) or (gate[1] not in self.list_reuse_qubit[layer])):
                    dict_qubit_interaction[gate[0]].append(gate[1])
                if gate[1] in dict_qubit_interaction and ((not test_reuse) or (gate[0] not in self.list_reuse_qubit[layer])):
                    dict_qubit_interaction[gate[1]].append(gate[0])
        expand_factor = 1

        # print("reuse qubit")
        # print(self.list_reuse_qubit[layer])
        # print("qubit to be placed:")
        # print(qubit_to_place)
        # # print("initial mapping: ")
        # # print(qubit_mapping)
        # # print("gate mapping:")
        # # print(self.mapping[0])
        # print("qubit interaction:")
        # print(dict_qubit_interaction)
    
        # construct the sparse matrix A for scipy to find minimal weight matching
        # use list_row_coo, list_col_coo, and data to represent A where A[list_i[k], list_j[k]] = data[k]
        site_storage_to_idx = dict()
        list_storage = []
        list_row_coo = []
        list_col_coo = []
        list_data = []
        # store data in coo form
        # row: rydberg site
        # col: gate id

        for i, q in enumerate(qubit_to_place):
            # add qubit's original location to the site candidate
            # print("generate candidate location for qubit {}".format(q))
            dict_bouding_box = dict()
            slm = self.architecture.dict_SLM[qubit_mapping[q][0]]
            lower_row = qubit_mapping[q][1]
            upper_row = qubit_mapping[q][1]
            left_col = qubit_mapping[q][2] 
            right_col = qubit_mapping[q][2]
            exact_loc_q = self.architecture.exact_SLM_location_tuple(qubit_mapping[q])
            exact_loc_gate = self.architecture.exact_SLM_location_tuple(self.mapping[0][q])
            if exact_loc_gate[1] < exact_loc_q[1]:
                lower_row = 0
            else:
                upper_row = slm.n_r

            # find the bounding box of the set of points q and the other qubits that have two-qubit gates with q
            slm = self.architecture.dict_SLM[qubit_mapping[q][0]]
            dict_bouding_box[qubit_mapping[q][0]] = [lower_row, upper_row, left_col, right_col] 
            for neighbor_q in dict_qubit_interaction[q]:
                tmp_slm_idx = last_gate_mapping[neighbor_q][0] 
                is_storage_slm = (self.architecture.dict_SLM[tmp_slm_idx].entanglement_id == -1)
                if is_storage_slm:
                    neighbor_q_location = last_gate_mapping[neighbor_q]
                else:
                    neighbor_q_location = self.architecture.nearest_storage_site(last_gate_mapping[neighbor_q][0], last_gate_mapping[neighbor_q][1], last_gate_mapping[neighbor_q][2])

                if neighbor_q_location[0] in dict_bouding_box:
                    dict_bouding_box[neighbor_q_location[0]][0] = min(neighbor_q_location[1], dict_bouding_box[neighbor_q_location[0]][0])
                    dict_bouding_box[neighbor_q_location[0]][1] = max(neighbor_q_location[1], dict_bouding_box[neighbor_q_location[0]][1])
                    dict_bouding_box[neighbor_q_location[0]][2] = min(neighbor_q_location[2], dict_bouding_box[neighbor_q_location[0]][2])
                    dict_bouding_box[neighbor_q_location[0]][3] = max(neighbor_q_location[2], dict_bouding_box[neighbor_q_location[0]][3])
                else:
                    slm = self.architecture.dict_SLM[neighbor_q_location[0]]
                    lower_row = neighbor_q_location[1]
                    upper_row = neighbor_q_location[1]
                    left_col = neighbor_q_location[2] 
                    right_col = neighbor_q_location[2]
                    exact_loc_neightbor_q = self.architecture.exact_SLM_location_tuple(neighbor_q_location)
                    if exact_loc_gate[1] < exact_loc_neightbor_q[1]:
                        lower_row = 0
                    else:
                        upper_row = slm.n_r
                    dict_bouding_box[neighbor_q_location[0]] = [lower_row, upper_row, left_col, right_col] 
            # consider the nearest storage site for the rydberg
            gate_location = last_gate_mapping[q]
            nearest_storage_site = self.architecture.nearest_storage_site(gate_location[0], gate_location[1], gate_location[2])
            ratio = 3
            if nearest_storage_site[0] in dict_bouding_box:
                dict_bouding_box[nearest_storage_site[0]][0] = min(nearest_storage_site[1] - ratio, dict_bouding_box[nearest_storage_site[0]][0])
                dict_bouding_box[nearest_storage_site[0]][1] = max(nearest_storage_site[1] + ratio, dict_bouding_box[nearest_storage_site[0]][1])
                dict_bouding_box[nearest_storage_site[0]][2] = min(nearest_storage_site[2] - ratio, dict_bouding_box[nearest_storage_site[0]][2])
                dict_bouding_box[nearest_storage_site[0]][3] = max(nearest_storage_site[2] + ratio, dict_bouding_box[nearest_storage_site[0]][3])
            else:
                dict_bouding_box[nearest_storage_site[0]] = [nearest_storage_site[1] - ratio, nearest_storage_site[1] + ratio,\
                                                         nearest_storage_site[2] - ratio, nearest_storage_site[2] + ratio]
            # print("bound box:")
            # print(dict_bouding_box)

            # according to the bounding box, find the valid location
            set_nearby_site = deepcopy(common_site)
            if is_empty_storage_site[qubit_mapping[q][0]][qubit_mapping[q][1]][qubit_mapping[q][2]]:
                set_nearby_site.add(qubit_mapping[q])
            
            for slm_id in dict_bouding_box:
                slm = self.architecture.dict_SLM[slm_id]
                bounding_box = dict_bouding_box[slm_id]
                bounding_box[0] = max(bounding_box[0] - expand_factor, 0)
                bounding_box[1] = min(bounding_box[1] + expand_factor + 1, slm.n_r)
                bounding_box[2] = max(bounding_box[2] - expand_factor, 0)
                bounding_box[3] = min(bounding_box[3] + expand_factor + 1, slm.n_c)
                for r in range(bounding_box[0], bounding_box[1]):
                    for c in range(bounding_box[2], bounding_box[3]):
                        if slm_id not in is_empty_storage_site or is_empty_storage_site[slm_id][r][c]:
                            set_nearby_site.add((slm_id, r, c))
            
            # print("possible sites:")
            # print(set_nearby_site)
            # input()
            
            for site in set_nearby_site:
                if site not in site_storage_to_idx:
                    site_storage_to_idx[site] = len(list_storage)
                    list_storage.append(site)
                idx_storage = site_storage_to_idx[site]
                # calculate cost 
                dis = self.architecture.distance(gate_location[0], gate_location[1], gate_location[2], site[0], site[1], site[2])
                lookahead_cost = 0
                for neighbor_q in dict_qubit_interaction[q]:
                    site_neighbor_q = last_gate_mapping[neighbor_q]
                    if self.architecture.dict_SLM[site_neighbor_q[0]].entanglement_id == -1:
                        # encourage them to be put in the same row
                        lookahead_cost += self.architecture.nearest_entanglement_site_dis(site[0], site[1], site[2], site_neighbor_q[0], site_neighbor_q[1], site_neighbor_q[2])
                    else:
                        exact_loc_neightbor_q = self.architecture.exact_SLM_location_tuple(last_gate_mapping[neighbor_q])
                        lookahead_cost += math.sqrt(math.dist(exact_loc_neightbor_q, exact_loc_q))
                cost = math.sqrt(dis) + 0.1 * lookahead_cost
                list_row_coo.append(idx_storage)
                list_col_coo.append(i)
                list_data.append(cost)
            # input()

        np_data = np.array(list_data)
        np_col_coo = np.array(list_col_coo)
        np_row_coo = np.array(list_row_coo)
        matrix = coo_matrix((np_data, (np_row_coo, np_col_coo)), shape=(len(list_storage), len(qubit_to_place)))
        # solve minimal matching by scipy
        site_ind, qubit_ind = min_weight_full_bipartite_matching(matrix)
        cost = matrix.toarray()[site_ind, qubit_ind].sum()
        if self.print_detail:
            print("[INFO]               Qubit placement cost: {}".format(cost))
        # process the solution
        tmp_mapping = deepcopy(last_gate_mapping)
        assert(len(qubit_to_place) == len(site_ind))
        for idx_storage, idx_qubit in zip(site_ind, qubit_ind):
            tmp_mapping[qubit_to_place[idx_qubit]] = list_storage[idx_storage]
            # qubit_mapping[qubit_to_place[idx_qubit]] = list_storage[idx_storage]
        self.mapping.append(tmp_mapping)


        # raise NotImplementedError