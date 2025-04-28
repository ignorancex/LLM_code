import time
from zac.placer.saplacer import SAPlacer
from zac.placer.vmplacer import VertexMatchingPlacer
from copy import deepcopy
from random import shuffle, seed
# from memory_profiler import profile

class Placer_mixin:
    """class to find a qubit layout via SA."""
    def place_qubit_initial(self):
        """
        generate qubit initial layout
        """
        t_p = time.time()
        if self.given_initial_mapping is not None:
            self.qubit_mapping.append(self.given_initial_mapping)
        else:
            # qubit placement for layout
            if self.trivial_placement:
                self.place_trivial()
                # !!!!
                # self.place_trivial_square()
                # return
            else:
                sa_placer = SAPlacer(self.l2)
                sa_placer.run(self.architecture, self.n_q, self.gate_scheduling)
                self.qubit_mapping.append(sa_placer.best_mapping)
        self.runtime_analysis["initial placement"] = time.time()- t_p
    
    def place_trivial(self):
        seed(0)
        slm_idx = 0
        slm = self.architecture.dict_SLM[self.architecture.storage_zone[slm_idx]]
        n_c = slm.n_c
        c = 0
        r = 0
        list_possible_position = []
        # decide begin with row 0 or row n
        dis1 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], r, c)
        dis2 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], slm.n_r - 1, c)
        if dis1 < dis2:
            step = 1
        else:
            r = slm.n_r - 1
            step = -1
        for i in range(self.n_q):
            list_possible_position.append((slm_idx, r, c))
            c += 1
            if c % n_c == 0:
                r += step
                c = 0
                if r == slm.n_r:
                    slm_idx += 1
                    slm = self.architecture.dict_SLM[self.architecture.storage_zone[slm_idx]]
                    if step > 0:
                        r = slm.n_r -1
                    else:
                        r = 0
                    n_c = slm.n_c
        # shuffle(list_possible_position)
        self.qubit_mapping.append(deepcopy(list_possible_position))
        # print(self.qubit_mapping)
    def place_trivial_square(self):
        slm_idx = 0
        slm = self.architecture.dict_SLM[self.architecture.storage_zone[slm_idx]]
        n_c = 61
        begin_c = 40
        c = begin_c
        r = 0
        list_possible_position = []
        # decide begin with row 0 or row n
        dis1 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], r, c)
        dis2 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], slm.n_r - 1, c)
        if dis1 < dis2:
            step = 1
        else:
            r = slm.n_r - 1
            step = -1
        for i in range(self.n_q):
            list_possible_position.append((slm_idx, r, c))
            c += 1
            if c % n_c == 0:
                r += step
                c = begin_c
                if r == slm.n_r:
                    slm_idx += 1
                    slm = self.architecture.dict_SLM[self.architecture.storage_zone[slm_idx]]
                    if step > 0:
                        r = slm.n_r -1
                    else:
                        r = 0
                    n_c = slm.n_c
        # shuffle(list_possible_position)
        self.qubit_mapping.append(deepcopy(list_possible_position))
        # print(self.qubit_mapping)
    # @profile
    def place_qubit_intermedeiate(self):
        """
        generate qubit initial layout
        """
        t_p = time.time()
        intermediate_placer = VertexMatchingPlacer(deepcopy(self.qubit_mapping[0]))
        # print("self.reuse_qubit")
        # print(self.reuse_qubit)
        intermediate_placer.run(self.architecture, self.qubit_mapping, self.gate_scheduling, self.dynamic_placement, self.reuse_qubit)
        self.qubit_mapping = intermediate_placer.mapping
        # for mapping in self.qubit_mapping:
        #     print(mapping)
        # for i in range(len(self.gate_scheduling)):
            # find the location of gates
            # print("qubit_mapping_in_storage")
            # print(qubit_mapping_in_storage)
            # print("self.gate_scheduling")
            # print(self.gate_scheduling[i])
            # qubit_mapping = intermediate_placer.run(self.architecture, self.qubit_mapping[-2:], self.gate_scheduling, i, self.dynamic_placement, self.reuse_qubit, qubit_mapping_in_storage)
            # self.qubit_mapping.append(intermediate_placer.mapping[-2].deepcopy())
            # self.qubit_mapping.append(intermediate_placer.mapping[-1].deepcopy())
            # print("new mapping")
            # print(intermediate_placer.mapping[-2])
            # print(intermediate_placer.mapping[-1])
            # print("qubit mapping")
            # for mapping in self.qubit_mapping:
            #     print(mapping)
            # input()
            # put qubits back to the storage zone after Rydberg stages  

        self.runtime_analysis["intermediate placement"] = time.time()- t_p
    
