import sys
import math
from random import randrange, shuffle, uniform, seed
from zac.ds.architecture import Architecture
from copy import deepcopy

class SAPlacer:
    """class to find a qubit layout via SA."""

    def __init__(self, l2: bool = False):
        seed(0)
        self.initialize_param()
        self.n_qubit = 0
        self.list_gate = []
        self.movement = []
        self.list_qubit_dict_gate = []
        self.l2 = l2

    def initialize_param(self):
        # sa related parameter    
        self.sa_t = 100000.0
        self.sa_t1 = 4.0
        self.sa_t_frozen = 0.000001
        self.sa_p = 0.987 # the initial probability to accept up-hill so-
        self.sa_l  = 400 
        self.sa_n = 0  
        self.sa_k = 7 # use for updating t
        self.sa_c = 100 # use for updating t
        self.sa_iter_limit = 1000

        self.sa_init_perturb_num = 100
        self.sa_uphill_avg_cnt = 0
        self.sa_uphill_sum = 0
        self.sa_delta_cost_cnt = 0
        self.sa_delta_sum = 0
        self.sa_delta = 0
        self.sa_n_trials = 1
        self.best_mapping = None
        self.best_cost = sys.maxsize
        self.current_mapping = None
        self.current_mapping_physical_to_program = None
        self.current_cost = sys.maxsize
        self.current_violation = 0
        self.tmp_violation = 0

    def preprocessing(self):
        # construct a list to store the gates acting each qubit
        max_level = 5
        self.list_weight = [1 - 0.1 * l for l in range(max_level)]
        self.list_qubit_dict_gate = [dict() for i in range(self.n_qubit)]
        # list_qubit_n_gate = [0 for i in range(self.n_qubit)]
        for i, gates in enumerate(self.list_gate):
            if i < max_level: 
                weight = self.list_weight[i]
            else:
                weight = self.list_weight[-1]
            for gate in gates:
                if gate[1] in self.list_qubit_dict_gate[gate[0]]:
                    self.list_qubit_dict_gate[gate[0]][gate[1]] += weight
                else:
                    self.list_qubit_dict_gate[gate[0]][gate[1]] = weight                
                self.list_qubit_dict_gate[gate[1]][gate[0]] = self.list_qubit_dict_gate[gate[0]][gate[1]]

    def run(self, arch: Architecture, n_qubit: int, list_gate: list):
        """ 
        Run SA to find a good mapping
        """
        print("[INFO] ZAC: SA-based placement")
        self.initialize_param()
        self.n_qubit = n_qubit
        self.architecture = arch
        self.list_gate = list_gate
        self.preprocessing()
        # large iteration
        for trial in range(self.sa_n_trials):
            self.init_sa_solution()
            print("[INFO] ZAC: SA-Based Placer: Iter {}, initial cost: {:4f}".format(trial, self.best_cost));  
            n_reject = 0
            self.sa_n = 0;
            while self.sa_t > self.sa_t_frozen:
                self.sa_n += 1
                self.sa_delta_cost_cnt = 0
                self.sa_delta_sum = 0
                n_reject = 0
                for i in range(self.sa_l):
                    self.make_movement() # make movement and calculate cost difference
                    self.sa_delta_cost_cnt += 1
                    self.sa_delta_sum += abs(self.sa_delta)
                    if self.sa_delta <= 0:
                        self.current_cost += self.sa_delta
                        if self.best_cost - self.current_cost > 1e-9:
                            self.update_optimal_sol()
                    else:   # delta > 0
                        if self.accept_worse_sol():
                            self.current_cost += self.sa_delta
                        else:
                            # undo current movement
                            self.recover()
                            n_reject += 1
                self.update_temperature()
                if self.sa_n > self.sa_iter_limit:
                    break
            print("[INFO] ZAC: SA-Based Placer: Iter {}, cost: {:4f}".format(trial, self.best_cost));  
        # final refinement
        # print("qubit mapping")
        # print(self.best_mapping)
        # for i in range(self.n_qubit):
        #     for x in range(self.chip_dim[0]):
        #         for y in range(self.chip_dim[1]):
        #             self.movement = (i, x, y)
        #             self.make_movement(True)
        #             self.current_cost += self.sa_delta
        #             if self.best_cost - self.current_cost > 1e-9:
        #                 self.update_optimal_sol()
        #             else:
        #                 self.recover()
        # print("[INFO] SA-Based Placer: Final cost: {:4f}".format(self.best_cost)) 

    def init_sa_solution(self):
        """ 
        generate a placement solution 
        """
        self.current_mapping = []
        self.current_mapping_physical_to_program = dict()

        # decide begin with row 0 or row n
        list_possible_position = []
        slm_idx = 0
        slm = self.architecture.dict_SLM[self.architecture.storage_zone[slm_idx]]
        n_c = slm.n_c
        r = 0
        c = 0
        dis1 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], r, c)
        dis2 = self.architecture.nearest_entanglement_site_distance(self.architecture.storage_zone[slm_idx], slm.n_r - 1, c)
        if dis1 < dis2:
            step = 1
        else:
            r = slm.n_r - 1
            step = -1
        for i in range(self.n_qubit):
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

        for slm_id in self.architecture.storage_zone:
            slm = self.architecture.dict_SLM[slm_id]
            self.current_mapping_physical_to_program[slm_id] = [[-1 for j in range(slm.n_c)] for i in range(slm.n_r)]

        cost_trivial = 0
        cost_shuffle = 0
        cost_shuffle2 = 0
        list_shuffle_postion = deepcopy(list_possible_position)
        shuffle(list_shuffle_postion)
        list_shuffle_postion2 = deepcopy(list_shuffle_postion)
        shuffle(list_shuffle_postion2)
        for i in range(self.n_qubit):
            for j in range(i):
                if j in self.list_qubit_dict_gate[i]:
                    dis = self.distance(list_possible_position[i], list_possible_position[j]) 
                    cost_trivial += (dis * self.list_qubit_dict_gate[i][j])
                    dis = self.distance(list_shuffle_postion[i], list_shuffle_postion[j]) 
                    cost_shuffle += (dis * self.list_qubit_dict_gate[i][j])
                    dis = self.distance(list_shuffle_postion2[i], list_shuffle_postion2[j]) 
                    cost_shuffle2 += (dis * self.list_qubit_dict_gate[i][j])
        
        if cost_trivial < cost_shuffle and cost_trivial < cost_shuffle2:
            self.current_cost = cost_trivial
            for i in range(self.n_qubit):
                self.current_mapping.append(list_possible_position[i])
                self.current_mapping_physical_to_program[list_possible_position[i][0]][list_possible_position[i][1]][list_possible_position[i][2]] = i
        elif cost_shuffle < cost_trivial and cost_shuffle < cost_shuffle2:
            self.current_cost = cost_shuffle
            for i in range(self.n_qubit):
                self.current_mapping.append(list_shuffle_postion[i])
                self.current_mapping_physical_to_program[list_shuffle_postion[i][0]][list_shuffle_postion[i][1]][list_shuffle_postion[i][2]] = i
        else:
            self.current_cost = cost_shuffle2
            for i in range(self.n_qubit):
                self.current_mapping.append(list_shuffle_postion2[i])
                self.current_mapping_physical_to_program[list_shuffle_postion2[i][0]][list_shuffle_postion2[i][1]][list_shuffle_postion2[i][2]] = i

        # site_begin_index = [0]
        # list_possible_position = [i for i in range(site_begin_index[-1])]
        # for slm_id in self.architecture.storage_zone:
        #     slm = self.architecture.dict_SLM[slm_id]
        #     self.current_mapping_physical_to_program[slm_id] = [[-1 for j in range(slm.n_c)] for i in range(slm.n_r)]
        #     list_possible_position += [i for i in range(site_begin_index[-2], site_begin_index[-1])]
        
        # random generate a placement
        # shuffle(list_possible_position)

        # for i in range(self.n_qubit):
        #     slm_id = 0
        #     while list_possible_position[i] >= site_begin_index[slm_id]:
        #         slm_id += 1
        #     slm_id -= 1
        #     pos_id = list_possible_position[i] - site_begin_index[slm_id]
        #     r = pos_id // self.architecture.dict_SLM[slm_id].n_c
        #     c = pos_id % self.architecture.dict_SLM[slm_id].n_c
        #     self.current_mapping.append((slm_id,r,c))
        #     self.current_mapping_physical_to_program[slm_id][r][c] = i
        
        # self.current_mapping = [(randrange(self.chip_dim[0]), randrange(self.chip_dim[1])) for i in self.n_q]

        if self.best_cost - self.current_cost > 1e-9:
            self.update_optimal_sol()

        # Initial Perturbation 
        self.init_perturb()
        # Initialize 
        self.sa_uphill_sum = 0
        self.sa_uphill_avg_cnt = 0
        self.sa_delta_sum = 0
        self.sa_delta_cost_cnt = 0
    
    def init_perturb(self):
        uphill_sum = 0
        sum_cost = 0
        uphill_cnt = 0
        for i in range(self.sa_init_perturb_num):
            self.make_movement()
            
            self.current_cost += self.sa_delta
            if self.best_cost - self.current_cost > 1e-9:
                self.update_optimal_sol()
            
            if self.sa_delta > 0:
                uphill_sum += self.sa_delta;
                uphill_cnt += 1
            
            sum_cost += self.current_cost
        self.sa_t1 = (uphill_sum / uphill_cnt) / ((-1)*math.log(self.sa_p))
        self.sa_t = self.sa_t1

    def accept_worse_sol(self):
        accept = (uniform(0, 1)) <= math.exp(-(self.sa_delta)/(self.sa_t))
        return accept

    def update_temperature(self):
        if self.sa_n <= self.sa_k:
            self.sa_t = (self.sa_t1 * abs(self.sa_delta_sum) / self.sa_delta_cost_cnt) / self.sa_n / self.sa_c
        else:
            self.sa_t = (self.sa_t1 * abs(self.sa_delta_sum) / self.sa_delta_cost_cnt) / self.sa_n
    
    def distance(self, pos1, pos2):
        dis = math.sqrt(self.architecture.nearest_entanglement_site_dis(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]))
        # dis += self.architecture.distance(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])
        if self.l2:
            dis = pow(dis, 2)
        return dis

    def make_movement(self, given_movement = False):
        # get movement
        if given_movement:
            qubit_to_move = self.movement[0]
            new_slm = self.movement[1]
            new_r = self.movement[2]
            new_c = self.movement[3]
            old_slm = self.current_mapping[qubit_to_move][0]
            old_r = self.current_mapping[qubit_to_move][1]
            old_c = self.current_mapping[qubit_to_move][2]
        else:
            qubit_to_move = randrange(self.n_qubit)
            # qubit_to_swap = qubit_to_move
            # while qubit_to_swap == qubit_to_move:
            #     qubit_to_swap = randrange(self.n_qubit)
            # new_slm = self.current_mapping[qubit_to_swap][0]
            # new_r = self.current_mapping[qubit_to_swap][1]
            # new_c = self.current_mapping[qubit_to_swap][2]
            old_slm = self.current_mapping[qubit_to_move][0]
            old_r = self.current_mapping[qubit_to_move][1]
            old_c = self.current_mapping[qubit_to_move][2]
            new_slm = randrange(len(self.architecture.storage_zone))
            move_disr = 0
            move_disc = 0
            while move_disr == 0 and move_disc == 0:
                if randrange(0, 1, 1) == 1:
                    move_disr = randrange(-5, 6, 1)
                move_disc = randrange(-5, 6, 1)
            new_r = max(0, min(old_r + move_disr, self.architecture.dict_SLM[new_slm].n_r - 1))
            new_c = max(0, min(old_r + move_disc, self.architecture.dict_SLM[new_slm].n_c - 1))
            # new_r = randrange(self.architecture.dict_SLM[new_slm].n_r)
            # new_c = randrange(self.architecture.dict_SLM[new_slm].n_c)
        qubit_be_affected = self.current_mapping_physical_to_program[new_slm][new_r][new_c]
        self.movement = (qubit_to_move, new_slm, new_r, new_c, old_slm, old_r, old_c)

        # calculate original cost
        set_affected_gate = set()
        for gate_qubit in self.list_qubit_dict_gate[qubit_to_move]:
            weight = self.list_qubit_dict_gate[qubit_to_move][gate_qubit]
            if gate_qubit < qubit_to_move:
                set_affected_gate.add((qubit_to_move, gate_qubit, weight))
            else:
                set_affected_gate.add((gate_qubit, qubit_to_move, weight))
        if qubit_be_affected > -1:
            for gate_qubit in self.list_qubit_dict_gate[qubit_be_affected]:
                weight = self.list_qubit_dict_gate[qubit_be_affected][gate_qubit]
                if gate_qubit < qubit_be_affected:
                    set_affected_gate.add((qubit_be_affected, gate_qubit, weight))
                else:
                    set_affected_gate.add((gate_qubit, qubit_be_affected, weight))
        
        ori_cost = 0
        for gate in set_affected_gate:
            q0 = gate[0]
            q1 = gate[1]
            weight = gate[2]
            # print("gate: q0: {}, q1: {}".format(q0, q1))
            # print("qubit mapping: q0: {}, q1: {}".format(self.current_mapping[q0], self.current_mapping[q1]))
            dis = self.distance(self.current_mapping[q0], self.current_mapping[q1]) 
            # print("dis: ", dis)
            ori_cost += (weight * dis)
        # print("ori_cost: ", ori_cost)
        
        # make movement
        self.current_mapping[qubit_to_move] = (new_slm, new_r, new_c)
        self.current_mapping_physical_to_program[new_slm][new_r][new_c] = qubit_to_move
        self.current_mapping_physical_to_program[old_slm][old_r][old_c] = qubit_be_affected
        if qubit_be_affected > -1:
            self.current_mapping[qubit_be_affected] = (old_slm, old_r, old_c)
        
        # calculate new cost
        new_cost = 0
        for gate in set_affected_gate:
            q0 = gate[0]
            q1 = gate[1]
            weight = gate[2]
            # print("gate: q0: {}, q1: {}".format(q0, q1))
            # print("qubit mapping: q0: {}, q1: {}".format(self.current_mapping[q0], self.current_mapping[q1]))
            dis = self.distance(self.current_mapping[q0], self.current_mapping[q1]) 
            # print("dis: ", dis)
            new_cost += (weight * dis)
        # print("new_cost: ", new_cost)

        self.sa_delta = new_cost - ori_cost

    def recover(self):
        qubit_to_move, old_slm, old_r, old_c, new_slm, new_r, new_c = self.movement
        qubit_be_affected = self.current_mapping_physical_to_program[new_slm][new_r][new_c]
        self.current_mapping[qubit_to_move] = (new_slm, new_r, new_c)
        self.current_mapping_physical_to_program[new_slm][new_r][new_c] = qubit_to_move
        self.current_mapping_physical_to_program[old_slm][old_r][old_c] = qubit_be_affected
        if qubit_be_affected > -1:
            self.current_mapping[qubit_be_affected] = (old_slm, old_r, old_c)
    
    def get_cost(self):
        cost = 0
        for i in range(self.n_qubit):
            for j in range(i):
                if j in self.list_qubit_dict_gate[i]:
                    dis = self.distance(self.current_mapping[i], self.current_mapping[j]) 
                    cost += (dis * self.list_qubit_dict_gate[i][j])
        return cost

    def update_optimal_sol(self):
        self.best_mapping = deepcopy(self.current_mapping)
        self.best_cost = self.current_cost

