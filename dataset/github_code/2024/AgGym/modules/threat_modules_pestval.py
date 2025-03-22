import copy
import numpy as np
import os
import sys
sys.path.append('/data/mahsak/AgGym2')
import random
import logging
from typing import Tuple, List
from utils import general_utils as gu
from multiprocessing import Pool, shared_memory, cpu_count
import time
import pdb

print("abc")
EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

gu.seed_everything(1337)

class Threat:
    def __init__(self, threat_type: str, severity: int, plot_size: Tuple[int, int], pesticide_action: dict):
        self.threat_type = threat_type
        self.severity = severity
        self.plot_size = plot_size
        self.key = False
        self.destruction_limit = [7,8,9]
        # Dead=[0.6, 1]
        self.pesticide_actions = pesticide_action

        logging.info("---------- Threat initiated ----------")
        logging.info(f"Name: {self.__class__.__name__}, Type: {self.threat_type}, Severity: {self.severity}")
        logging.info(f"Destruction days: {self.destruction_limit}")
        logging.info("---------- Threat log end ----------")

    def reset(self):
        self.infect_list = {}
        self.key = False

    def apply_infection_reproductive_stage(self):
        if self.gs_title.find('R') != -1 and self.key == False:
            self.key = True
            random_init_x = np.random.randint(0, self.plot_size[1])
            random_init_y = np.random.randint(0, self.plot_size[0])
            self.infect_list = {(random_init_y, random_init_x):0}
            for infect in self.infect_list:
                y, x = infect
                self.plot_state[y, x] = INFECTED

    def apply_infection_reproductive_stage_survival(self):
        infect_count = np.count_nonzero(self.plot_state == INFECTED)
        dead_count = np.count_nonzero(self.plot_state == DEAD)

        if self.plot_size[0] * self.plot_size[1] == infect_count + dead_count:
            pass
        else:
            infect_key = np.random.choice([0,1,2,3,4,5,6,7,8,9])

            if infect_key == 0:
                random_init_x = np.random.randint(0, self.plot_size[1])
                random_init_y = np.random.randint(0, self.plot_size[0])
                while True:
                    if (random_init_y, random_init_x) in self.infect_list or self.plot_state[random_init_y, random_init_x] == DEAD:
                        random_init_x = np.random.randint(0, self.plot_size[1])
                        random_init_y = np.random.randint(0, self.plot_size[0])
                    else:
                        break
                self.infect_list[(random_init_y, random_init_x)] = 0
                self.plot_state[random_init_y, random_init_x] = INFECTED

    def apply_pesticide(self, action):
        logging.debug(f"Infect list before pest: {self.infect_list}")
        pesticide_chance = self.pesticide_actions[action]
        temp_list = copy.deepcopy(self.infect_list)
        pest_chance_list = np.random.choice([0,1], size=len(temp_list), replace=True, p=[1-pesticide_chance, pesticide_chance])
        for p, i in zip(pest_chance_list, temp_list.keys()):
            if p == 1:
                y, x = i
                # if not dead yet, cure it
                if self.plot_state[y, x] != DEAD:
                    # print(f"Applying pesticide on {x} {y}")
                    self.plot_state[y, x] = ALIVE
                    del self.infect_list[i]

    def spread_probabilities(self):
        high_infect = np.random.choice([0,1], size=4, replace=True, p=[0.7, 0.3])
        medium_infect = np.random.choice([0,1], size=4, replace=True, p=[0.85, 0.15])
        low_infect = np.random.choice([0,1], size=8, replace=True, p=[0.95, 0.05])
        self.infection = {'high': high_infect, 'medium': medium_infect, 'low': low_infect}

    def infection_tracker(self):
        # +1 day for infect survival, if #days reached destruction limit, mark plot state as DEAD
        temp_list = copy.deepcopy(self.infect_list)
        for k in temp_list.keys():
            y, x = k
            self.infect_list[k] += 1
            limit = np.random.choice(self.destruction_limit)
            if self.infect_list[k] >= limit:
                self.plot_state[y, x] = DEAD

                neigh_list = []

                # check if neighbouring dead have spread
                # check top
                if y-1 >= 0:
                    if (y-1, x) not in self.infect_list.keys():
                        neigh_list.append((y-1, x))

                #check bottom
                if y+1 < self.plot_size[1]:
                    if (y+1, x) not in self.infect_list.keys():
                        neigh_list.append((y+1, x))

                #check left
                if x-1 >= 0:
                    if (y, x-1) not in self.infect_list.keys():
                        neigh_list.append((y, x-1))

                #check right
                if x+1 < self.plot_size[0]:
                    if (y, x+1) not in self.infect_list.keys():
                        neigh_list.append((y, x+1))

                #check top left
                if y-1 >= 0 and x-1 >= 0:
                    if (y-1, x-1) not in self.infect_list.keys():
                        neigh_list.append((y-1, x-1))

                #check top right
                if y-1 >= 0 and x+1 < self.plot_size[0]:
                    if (y-1, x+1) not in self.infect_list.keys():
                        neigh_list.append((y-1, x+1))

                #check bottom left
                if y+1 < self.plot_size[1] and x-1 >= 0:
                    if (y+1, x-1) not in self.infect_list.keys():
                        neigh_list.append((y+1, x-1))

                #check bottom right
                if y+1 < self.plot_size[1] and x+1 < self.plot_size[0]:
                    if (y+1, x+1) not in self.infect_list.keys():
                        neigh_list.append((y+1, x+1))

                if len(neigh_list) == 8:
                    index = np.random.choice(len(neigh_list))
                    self.plot_state[neigh_list[index]] = INFECTED
                    self.infect_list[neigh_list[index]] = 0

    def check_neighbour(self):
        # print(f"Infect list after pest: {self.infect_list}")
        temp_list = copy.deepcopy(self.infect_list)

        for plant_infected in temp_list.keys():
            # compute interaction
            # infection = self.compute_infection()
            y, x = plant_infected
            if self.plot_state[y, x] == DEAD:
                continue
            # print(f"For infected plant {x} {y}")
            # print(infection)

            #check top
            if y-1 >= 0:
                if self.infection['high'][0] == 1:
                    if (y-1, x) not in self.infect_list.keys():
                        # print('top')
                        self.plot_state[y-1, x] = INFECTED
                        self.infect_list[(y-1, x)] = 0

            #check bottom
            if y+1 < self.plot_size[1]:
                if self.infection['high'][1] == 1:
                    if (y+1, x) not in self.infect_list.keys():
                        # print('bottom')
                        self.plot_state[y+1, x] = INFECTED
                        self.infect_list[(y+1, x)] = 0

            #check left
            if x-1 >= 0:
                if self.infection['high'][2] == 1:
                    if (y, x-1) not in self.infect_list.keys():
                        # print('left')
                        self.plot_state[y, x-1] = INFECTED
                        self.infect_list[(y, x-1)] = 0

            #check right
            if x+1 < self.plot_size[0]:
                if self.infection['high'][3] == 1:
                    if (y, x+1) not in self.infect_list.keys():
                        # print('right')
                        self.plot_state[y, x+1] = INFECTED
                        self.infect_list[(y, x+1)] = 0

            #check top left
            if y-1 >= 0 and x-1 >= 0:
                if self.infection['medium'][0] == 1:
                    if (y-1, x-1) not in self.infect_list.keys():
                        # print('top left')
                        self.plot_state[y-1, x-1] = INFECTED
                        self.infect_list[(y-1, x-1)] = 0

            #check top right
            if y-1 >= 0 and x+1 < self.plot_size[0]:
                if self.infection['medium'][1] == 1:
                    if (y-1, x+1) not in self.infect_list.keys():
                        # print('top right')
                        self.plot_state[y-1, x+1] = INFECTED
                        self.infect_list[(y-1, x+1)] = 0

            #check bottom left
            if y+1 < self.plot_size[1] and x-1 >= 0:
                if self.infection['medium'][2] == 1:
                    if (y+1, x-1) not in self.infect_list.keys():
                        self.plot_state[y+1, x-1] = INFECTED
                        self.infect_list[(y+1, x-1)] = 0

            #check bottom right
            if y+1 < self.plot_size[1] and x+1 < self.plot_size[0]:
                if self.infection['medium'][3] == 1:
                    if (y+1, x+1) not in self.infect_list.keys():
                        # print('bottom right')
                        self.plot_state[y+1, x+1] = INFECTED
                        self.infect_list[(y+1, x+1)] = 0

    def compute_infection(self, action, plot_state, gs_title, sim_mode):
        self.gs_title = gs_title
        self.plot_state = plot_state
        if sim_mode == 'growthseason':
            self.apply_infection_reproductive_stage()
        elif sim_mode == 'survival':
            self.apply_infection_reproductive_stage_survival()
        else:
            assert 1 == 2, "Compute infection sim mode error exception"
        self.apply_pesticide(action)
        self.spread_probabilities()
        self.infection_tracker()
        self.check_neighbour()

        return self.plot_state


class Threat_basic:
    def __init__(self, w: int, h: int, ow: int, oh: int, 
                 grid: np.ndarray, coords_x: List[float], coords_y: List[float], pesticide_action: dict):
        # self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.ow, self.oh = ow, oh
        
        self.grid = grid
        self.infect_day_mat = np.zeros(self.grid.shape)
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.num_points = len(self.coords_x)
        self.pesticide_actions = pesticide_action
        self.destruction_limit = [13,14,15]
        self.infect_list = []
        self.Degraded_list=[]
        self.infect_list_before_pest = []
        self.grid_list=[]
        self.alive_sprayed=[]
        self.infect_sprayed=[]
        
        logging.info("---------- Threat initiated ----------")
        logging.info(f"Name: {self.__class__.__name__}")
        logging.info(f"Destruction days: {self.destruction_limit}")
        logging.info("---------- Threat log end ----------")

    def reset(self):
        self.infect_list = []
        self.infect_list_before_pest = []
        self.alive_sprayed=[]
        self.infect_sprayed=[]
        self.infect_day_mat = np.zeros(self.grid.shape)
        self.key = False

    def compute_dense(self, cx: int, cy: int, mat: np.ndarray, val: float = None):

        self.cx, self.cy = cx, cy
        self.west_edge, self.east_edge = int(cx - self.w/2), int(cx + self.w/2)
        self.north_edge, self.south_edge = int(cy + self.h/2), int(cy - self.h/2)
        
        if val == None:
            ty, tx = np.where(mat[self.south_edge:self.north_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.south_edge:self.north_edge+1, self.west_edge:self.east_edge+1] == val)
            
        ty = ty + self.south_edge
        tx = tx + self.west_edge

        if val != None:
            for y, x in zip(ty, tx):
                mat[y,x] = val

        return mat, (tx,ty)
    
    def compute_hollow(self, cx: int, cy: int, mat: np.ndarray, val: float = None):
        self.cx, self.cy = cx, cy
        # to prevent double counting from the compute_dense(), move the nsew off by 1
        #
        #    *  *  *  *  *
        #       /\
        #       |
        #       _
        #    *  *  *  *  *
        #     <-|     |->
        #    *  *  *  *  *
        #     <-|     |->
        #    *  *  *  *  *
        #       _
        #       |
        #       \/
        #    *  *  *  *  *
        #
        #
        self.west_edge, self.east_edge = int(cx - self.w/2) - 1, int(cx + self.w/2) + 1
        self.north_edge, self.south_edge = int(cy + self.h/2) + 1, int(cy - self.h/2) - 1
        self.o_west_edge, self.o_east_edge = int(cx - self.ow/2), int(cx + self.ow/2)
        self.o_north_edge, self.o_south_edge = int(cy + self.oh/2), int(cy - self.oh/2)
        
        if val == None:
            ty, tx = np.where(mat[self.north_edge:self.o_north_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.north_edge:self.o_north_edge+1, self.west_edge:self.east_edge+1] == val)

        ty1 = ty + self.north_edge
        tx1 = tx + self.west_edge

        if val != None:
            for y, x in zip(ty1, tx1):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.south_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.south_edge+1, self.west_edge:self.east_edge+1] == val)

        ty2 = ty + self.o_south_edge
        tx2 = tx + self.west_edge

        if val != None:
            for y, x in zip(ty2, tx2):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.o_west_edge:self.west_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.o_west_edge:self.west_edge+1] == val)

        ty3 = ty + self.o_south_edge
        tx3 = tx + self.o_west_edge

        if val != None:
            for y, x in zip(ty3, tx3):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.east_edge:self.o_east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.east_edge:self.o_east_edge+1] == val)

        ty4 = ty + self.o_south_edge
        tx4 = tx + self.east_edge

        if val != None:
            for y, x in zip(ty4, tx4):
                mat[y,x] = val

        tx = np.hstack((tx1, tx2, tx3, tx4))
        ty = np.hstack((ty1, ty2, ty3, ty4))
        
        return mat, (tx,ty)
    
    def draw(self, ax, c='k', lw=1, **kwargs):
        try:
            self.west_edge
        except AttributeError:
            return
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        
        x3, y3 = self.o_west_edge, self.o_north_edge
        x4, y4 = self.o_east_edge, self.o_south_edge
        
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)
        ax.plot([x3,x4,x4,x3,x3],[y3,y3,y4,y4,y3], c=c, lw=lw, **kwargs)

    def apply_infection_reproductive_stage(self):
        if (self.gs_title.find('R') != -1 or self.gs_title.find('StartFlowering') != -1) and self.key == False:
            self.key = True
            random_index = np.random.randint(0, self.num_points)
            random_init_x = self.coords_x[random_index]
            random_init_y = self.coords_y[random_index]
            self.infect_list = [(random_init_y, random_init_x)]
            for infect in self.infect_list:
                y, x = infect
                self.grid[y, x] = INFECTED
            # self.infect_list_before_pest=copy.deepcopy(self.infect_list)

    def apply_infection_reproductive_stage_survival(self):
        infect_count = np.count_nonzero(self.grid == INFECTED)
        dead_count = np.count_nonzero(self.grid == DEAD)

        if self.num_points == infect_count + dead_count:
            pass
        else:
            infect_key = np.random.choice([0,1,2,3,4,5,6,7,8,9])

            if infect_key == 0:
                random_index = np.random.randint(0, self.num_points)
                random_init_x = self.coords_x[random_index]
                random_init_y = self.coords_y[random_index]
                while True:
                    if (random_init_y, random_init_x) in self.infect_list or self.grid[random_init_y, random_init_x] == DEAD:
                        random_index = np.random.randint(0, self.num_points)
                        random_init_x = self.coords_x[random_index]
                        random_init_y = self.coords_y[random_index]
                    else:
                        break
                self.infect_list.append((random_init_y, random_init_x))
                self.grid[random_init_y, random_init_x] = INFECTED
    def reshape_to_max_square(vec):
        """Reshape a vector to the largest possible square matrix, padding with zeros."""
        
        # Find the dimension of the target matrix
        dim = int(np.ceil(np.sqrt(len(vec))))
        
        # Calculate how many zeros need to be added
        padding = (dim * dim) - len(vec)
        
        # Add the zeros to the end of the vector
        padded_vec = np.pad(vec, (0, padding))
        
        # Reshape the vector to the matrix form
        matrix = padded_vec.reshape(dim, dim)
        
        return matrix



    def apply_pesticide(self, action: float, withpest_val):
        logging.debug(f"Infect list before pest: {self.infect_list}")
        self.infect_list_before_pest=copy.deepcopy(self.infect_list)
        pesticide_chance = self.pesticide_actions[action]
        # self.alive_sprayed=[]
        self.grid_list=[]
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                self.grid_list.append((i, j))
        #Validation once having pesticide application across the entire field in a single day
        # if self.withpest_val == True:
        #     for i in self.grid_list:
        #         if not any(np.array_equal(i, infected) for infected in self.infect_list) and not any(np.array_equal(i, degraded) for degraded in self.Degraded_list):
        #             self.alive_sprayed.append(i)
        #     if action!=0:
        #         for i in self.infect_list_before_pest:
        #             if i not in self.infect_sprayed:
        #                 self.infect_sprayed.append(i)
        # else:
        #     if action!=0 and len(self.infect_list)==0 and len(self.alive_sprayed) == 0:
        #     #     action =0
        #     #     pesticide_chance = self.pesticide_actions[action]       
        #         self.alive_sprayed=self.grid_list
        # # if len(self.infect_list)!=0:
        #         for i in self.infect_list_before_pest:
        #             if i not in self.infect_sprayed:
        #                 self.infect_sprayed.append(i)
        if withpest_val=='True' and action != 0:
            # Add all non-infected and non-degraded plots that have been sprayed to alive_sprayed list
            self.alive_sprayed = [i for i in self.grid_list if not any(np.array_equal(i, infected) for infected in self.infect_list) and not any(np.array_equal(i, degraded) for degraded in self.Degraded_list)]
    
            # If action is non-zero, update infect_sprayed list with new infections that have not been sprayed
            # if action != 0:
            self.infect_sprayed.extend([i for i in self.infect_list_before_pest if i not in self.infect_sprayed])
        else:
            # If action is non-zero and there are no infections and no alive_sprayed, 
             # set alive_sprayed to all grid points and update infect_sprayed list
            if action != 0 and not self.infect_list and not self.alive_sprayed:
                self.alive_sprayed = self.grid_list.copy()
                self.infect_sprayed.extend([i for i in self.infect_list_before_pest if i not in self.infect_sprayed])


        temp_list = []
        pest_chance_list = np.random.choice([0,1], size=len(self.infect_list), replace=True, p=[1-pesticide_chance, pesticide_chance])
        # for p in pest_chance_list:
        #     if action!=0 and len(self.infect_list) == 0:
        #         self.alive_sprayed=self.grid_list
        # if pesticide_reward[self.action] != 0 and len(self.threat.infect_list) == 0:
        idx = 0
        # self.infect_sprayed=[]
        for p, i in zip(pest_chance_list, self.infect_list):
            if p == 1:
                y, x = i
                # if i not in self.infect_sprayed:
                #     self.infect_sprayed.append(i)
                # if not dead yet, cure it
                if self.grid[y, x] != DEAD:
                    # print(f"Applying pesticide on {x} {y}")
                    self.grid[y, x] = ALIVE
                    self.infect_day_mat[y, x] = 0.
                    temp_list.append(idx)
            idx += 1
        
        for i in temp_list[::-1]:
            self.infect_list.pop(i)
    
        
    def spread_probabilities(self):
        high_infect = np.random.choice([0, 1, 2], size=8, replace=True, p=[0.6, 0.25, 0.15])
        medium_infect = np.random.choice([0,1, 2], size=16, replace=True, p=[0.75, 0.15, 0.1])
        low_infect = np.random.choice([0,1], size=24, replace=True, p=[0.95, 0.05])
        self.infection = {'high': high_infect, 'medium': medium_infect, 'low': low_infect}
    def spread_probabilities_sprayed(self):
        high_infect = np.random.choice([0, 1, 2], size=8, replace=True, p=[0.75, 0.23, 0.02])
        medium_infect = np.random.choice([0,1, 2], size=16, replace=True, p=[0.8, 0.17, 0.03])
        low_infect = np.random.choice([0,1], size=24, replace=True, p=[0.95, 0.05])
        self.infection = {'high': high_infect, 'medium': medium_infect, 'low': low_infect}


    def infection_tracker(self, severity,  timestep):
        # +1 day for infect survival, if #days reached destruction limit, mark plot state as DEAD
        # start = time.time()
        self.timestep=timestep
        temp_list = list(self.infect_list)
        # end = time.time()
        # print(f"Infection Tracker temp_list | Time taken: {end - start}")
        dense_list = [0]
        loop_list = [0]
        apply_list = [0]
        start_loop = time.time()
        idx = 0
        self.Degraded_list=[]
        for k in temp_list:
            y, x = k
            self.infect_day_mat[y, x] += 1
            limit = np.random.choice(self.destruction_limit)
            if self.infect_day_mat[y, x] >= limit:
                if self.grid[y, x] != DEAD:
                    self.grid[y, x] = DEAD

                    if 46<=self.timestep<=56:
                        self.Degraded_list.append( (((np.exp(-self.infect_day_mat[y, x]+12) / (1+np.exp(-self.infect_day_mat[y, x]+12)))*0.7) - 0.7)*severity)
                    elif 57<=self.timestep<=74:
                        self.Degraded_list.append( (((np.exp(-self.infect_day_mat[y, x]+12) / (1+np.exp(-self.infect_day_mat[y, x]+12)))*0.6) - 0.6)*severity)
                    else:
                        self.Degraded_list.append( (((np.exp(-self.infect_day_mat[y, x]+12) / (1+np.exp(-self.infect_day_mat[y, x]+12)))*0.5) - 0.5)*severity)

                self.infect_day_mat[y, x] = 0.
                # dense_list.append(idx)
                #
                # check if need to remove dead grid from infect_list (afaik, seems like not removing it)
                #

                unaffected_neighbour_list = []

                # check if neighbouring dead have spread
                start = time.time()
                _, neighbouring_idx = self.compute_dense(x, y, self.grid)
                end = time.time()
                dense_list.append(end - start)

                for i, j in zip(neighbouring_idx[0], neighbouring_idx[1]):
                    if (j, i) not in self.infect_list:
                        unaffected_neighbour_list.append((j, i))

                end = time.time()
                loop_list.append(end - start)

                start = time.time()
                
                if len(unaffected_neighbour_list) == 8:
                    j, i = unaffected_neighbour_list[np.random.choice(len(unaffected_neighbour_list))]
                    self.grid[j, i] = INFECTED
                    self.infect_list.append((j, i))

                end = time.time()
                apply_list.append(end - start)

        end = time.time()
        logging.debug(f"Infection Tracker whole loop | Time taken: {end - start_loop}")
        logging.debug(f"Infection Tracker Dense | Total {sum(dense_list)}, Average {np.round(np.average(dense_list), 6)}, Max: {np.round(max(dense_list), 6)}, Min: {np.round(min(dense_list), 6)} ")
        logging.debug(f"Infection Tracker Loop | Total {sum(loop_list)}, Average {np.round(np.average(loop_list), 6)}, Max: {np.round(max(loop_list), 6)}, Min: {np.round(min(loop_list), 6)} ")
        logging.debug(f"Infection Tracker Apply | Total {sum(apply_list)}, Average {np.round(np.average(apply_list), 6)}, Max: {np.round(max(apply_list), 6)}, Min: {np.round(min(apply_list), 6)} ")

    def check_neighbour(self):
        temp_list = list(self.infect_list)
        
        dense_list = [0]
        dense_loop = [0]
        hollow_list = [0]
        hollow_loop = [0]

        start_loop = time.time()
        for plant_infected in temp_list:
            y, x = plant_infected
            if self.grid[y, x] == DEAD:
                continue
            
            start = time.time()
            _, neighbouring_idx = self.compute_dense(x, y, self.grid)
            end = time.time()
            dense_list.append(end - start)
            idx = 0
            start = time.time()
            for (i, j) in zip(neighbouring_idx[0], neighbouring_idx[1]):
                if (i, j) == (x, y):
                    continue
                if (j, i) not in self.infect_list and self.grid[j, i] != DEAD and len(self.alive_sprayed)==0:
                    if (j, i) not in self.infect_sprayed and self.infection['high'][idx] == 1:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))
                    # the probability of initially getting infected,
                    #  then being treated (sprayed) and recovering (becoming alive), followed by a subsequent reinfection
                    elif (j, i) in self.infect_sprayed and self.infection['high'][idx] == 2:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))
                    # elif len(self.infect_list) == 0:
                elif (j, i) not in self.infect_list and self.grid[j, i] != DEAD and len(self.alive_sprayed)!=0:
                    if (j, i) not in self.infect_sprayed and self.infection['high'][idx] == 1:
                            self.grid[j, i] = INFECTED
                            self.infect_list.append((j,i))
                    # The probability of an already immuned plot (alive) getting sprayed, 
                    # followed by infection, subsequent treatment (spraying), recovery (becoming alive again), 
                    # and then experiencing reinfection. This wouldnt occure in the actual scenario, where spraying occurs only once,
                    # so the probability should be set to zero once validating the real situation 
                    elif (j, i) in self.infect_sprayed and self.infection['high'][idx] == 2:
                            self.grid[j, i] = INFECTED
                            self.infect_list.append((j,i))
                idx += 1
            end = time.time()
            dense_loop.append(end - start)
                
            start = time.time()
            _, neighbouring_idx = self.compute_hollow(x, y, self.grid)
            end = time.time()
            hollow_list.append(end - start)
            idx = 0
            start = time.time()
            for i, j in zip(neighbouring_idx[0], neighbouring_idx[1]):
                if (i, j) == (x, y):
                    continue
                if (j, i) not in self.infect_list and self.grid[j, i] != DEAD and len(self.alive_sprayed)==0:
                    if (j, i) not in self.infect_sprayed and self.infection['medium'][idx] == 1:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))
                    elif (j, i) in self.infect_sprayed and self.infection['medium'][idx] == 2:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))
                elif (j, i) not in self.infect_list and self.grid[j, i] != DEAD and len(self.alive_sprayed)!=0:
                    if (j, i) not in self.infect_sprayed and self.infection['medium'][idx] == 1:
                            self.grid[j, i] = INFECTED
                            self.infect_list.append((j,i))
                    elif (j, i) in self.infect_sprayed and self.infection['medium'][idx] == 2:
                            self.grid[j, i] = INFECTED
                            self.infect_list.append((j,i))
                idx += 1
            end = time.time()
            hollow_loop.append(end - start)
        
        end = time.time()
        logging.debug(f"check_neighbour whole loop | Time taken: {end - start_loop}")
        logging.debug(f"check_neighbour Dense | Total {sum(dense_list)}, Average {np.round(np.average(dense_list), 6)}, Max: {np.round(max(dense_list), 6)}, Min: {np.round(min(dense_list), 6)} ")
        logging.debug(f"check_neighbour Loop | Total {sum(dense_loop)}, Average {np.round(np.average(dense_loop), 6)}, Max: {np.round(max(dense_loop), 6)}, Min: {np.round(min(dense_loop), 6)} ")
        logging.debug(f"check_neighbour Hollow | Total {sum(hollow_list)}, Average {np.round(np.average(hollow_list), 6)}, Max: {np.round(max(hollow_list), 6)}, Min: {np.round(min(hollow_list), 6)} ")
        logging.debug(f"check_neighbour Loop | Total {sum(hollow_loop)}, Average {np.round(np.average(hollow_loop), 6)}, Max: {np.round(max(hollow_loop), 6)}, Min: {np.round(min(hollow_loop), 6)} ")

    def infectedyield(self, severity):
        temp_list = list(self.infect_list)
        self.infectdeg_list=[]
        for k in temp_list:
            y, x = k
            self.infectdeg_list.append( (((np.exp(-self.infect_day_mat[y, x]+6) / (1+np.exp(-self.infect_day_mat[y, x]+6)))*0.6) - 0.6)*severity)
        return self.infectdeg_list
        
        
            
    def compute_infection(self, action, grid, gs_title, timestep, sim_mode, withpest_val, severity):
        self.gs_title = gs_title
        self.grid = grid
        
        if sim_mode == 'growthseason':
            start = time.time()
            self.apply_infection_reproductive_stage()
            end = time.time()
            logging.debug(f"Apply Infection | Time taken: {end - start}")
        elif sim_mode == 'survival':
            self.apply_infection_reproductive_stage_survival()
        else:
            assert 1 == 2, "Compute infection sim mode error exception"
        print(f"Infected before pesticide #={len(self.infect_list)}")
        start = time.time()
        self.apply_pesticide(action, withpest_val)
        end = time.time()
        print(f"Infected after pesticide #={len(self.infect_list)}")
        logging.debug(f"Apply Pesticide | Time taken: {end - start}")
        start = time.time()
        if len(self.alive_sprayed)==0:
            self.spread_probabilities()
        else:
            self.spread_probabilities_sprayed()
        end = time.time()
        logging.debug(f"Spread Probabilities | Time taken: {end - start}")
        start = time.time()
        self.infection_tracker(severity, timestep)
        end = time.time()
        logging.debug(f"Infection Tracker | Time taken: {end - start}")
        start = time.time()
        self.check_neighbour()
        end = time.time()
        logging.debug(f"Check Neighbour | Time taken: {end - start}")
        self.infectedyield(severity)
        return self.grid, self.Degraded_list, self.infectdeg_list
        

# # # # # # multiprocessing

class Threat_mp:
    def __init__(self, threat_type: str, severity: int, plot_size: Tuple[int, int], pesticide_action: dict):
        self.threat_type = threat_type
        self.severity = severity
        self.plot_size = plot_size
        self.key = False
        self.destruction_limit = [7,8,9]
        self.pesticide_actions = pesticide_action

        logging.info("---------- Threat initiated ----------")
        logging.info(f"Name: {self.__class__.__name__}, Type: {self.threat_type}, Severity: {self.severity}")
        logging.info(f"Destruction days: {self.destruction_limit}")
        logging.info("---------- Threat log end ----------")

    def init_shm(self):
        # Allocate shared memory
        print(f"Allocating shared memory")
        self.shm = shared_memory.SharedMemory(create=True, size=self.plot_state.nbytes)
        self.shm_array = np.ndarray(self.plot_state.shape, dtype=self.plot_state.dtype, buffer=self.shm.buf)
        self.shm_array[:] = self.plot_state[:]

    def rmv_shm(self):
        print(f"Deallocating shared memory")
        self.plot_state[:] = self.shm_array[:]
        self.shm.close()
        self.shm.unlink()

    def reset(self):
        self.infect_list = {}
        self.key = False

    def apply_infection_reproductive_stage(self):
        print(f"Applying infection")
        if self.gs_title.find('R') != -1 and self.key == False:
            self.key = True
            random_init_x = np.random.randint(0, self.plot_size[1])
            random_init_y = np.random.randint(0, self.plot_size[0])
            self.infect_list = {(random_init_y, random_init_x):0}
            self.plot_state[random_init_y, random_init_x] = INFECTED

    def apply_pesticide_mp_apply(self, chance, infect, shr_name):
        print(f"Applying pesticide. Process with {infect}")	
        if chance == 1:
            existing_shm = shared_memory.SharedMemory(name=shr_name)
            temp_array = np.ndarray(self.plot_size, dtype=np.float64, buffer=existing_shm.buf)
            y, x = infect
            # if not dead yet, cure it
            if temp_array[y, x] != DEAD:
                # print(f"Applying pesticide on {x} {y}")
                temp_array[y, x] = ALIVE
                # del self.infect_list[infect]
            existing_shm.close()
            return (y,x)
        else:
            return ()

    def apply_pesticide(self, action):
        logging.debug(f"Infect list before pest: {self.infect_list}")
        pesticide_chance = self.pesticide_actions[action]
        temp_list = copy.deepcopy(self.infect_list)
        pest_chance_list = np.random.choice([0,1], size=len(temp_list), replace=True, p=[1-pesticide_chance, pesticide_chance])
        
        p = Pool(cpu_count())
        x = []
        for i,j in zip(pest_chance_list, temp_list.keys()):
            x.append((i,j, self.shm.name))
        results = p.starmap(self.apply_pesticide_mp_apply, x)

        for i in results:
            if len(i) == 2:
                del self.infect_list[i]
        p.close()
        p.join()

    def spread_probabilities(self):
        high_infect = np.random.choice([0,1], size=4, replace=True, p=[0.7, 0.3])
        medium_infect = np.random.choice([0,1], size=4, replace=True, p=[0.85, 0.15])
        low_infect = np.random.choice([0,1], size=8, replace=True, p=[0.95, 0.05])
        self.infection = {'high': high_infect, 'medium': medium_infect, 'low': low_infect}
        # print(self.infection)

    def infection_tracker_mp_apply(self, infect, shr_name):
        print(f"Infection tracker. Process with {infect}")	
        existing_shm = shared_memory.SharedMemory(name=shr_name)
        temp_array = np.ndarray(self.plot_size, dtype=np.float64, buffer=existing_shm.buf)
        y, x = infect
        self.infect_list[infect] += 1
        limit = np.random.choice(self.destruction_limit)
        if self.infect_list[infect] >= limit:
            temp_array[y, x] = DEAD
        existing_shm.close()

        return (y,x)

    def infection_tracker(self):
        # +1 day for infect survival, if #days reached destruction limit, mark plot state as DEAD
        temp_list = copy.deepcopy(self.infect_list)

        p = Pool(cpu_count())
        x = []
        for i in temp_list.keys():
            x.append((i, self.shm.name))
        results = p.starmap(self.infection_tracker_mp_apply, x)

        for i in results:
            if len(i) == 2:
                self.infect_list[i] += 1
        p.close()
        p.join()

    def check_neighbour_mp_apply(self, infect, shr_name):
        print(f"Checking neighbour. Process with {infect}")	
        existing_shm = shared_memory.SharedMemory(name=shr_name)
        temp_array = np.ndarray(self.plot_size, dtype=np.float64, buffer=existing_shm.buf)
        y, x = infect
        if temp_array[y, x] == DEAD:
            existing_shm.close()
            return ()

        tp_list = []

        #check top
        if y-1 >= 0:
            if self.infection['high'][0] == 1:
                if (y-1, x) not in self.infect_list.keys():
                    temp_array[y-1, x] = INFECTED
                    # self.infect_list[(y-1, x)] = 0
                    tp_list.append((y-1, x))

        #check bottom
        if y+1 < self.plot_size[1]:
            if self.infection['high'][1] == 1:
                if (y+1, x) not in self.infect_list.keys():
                    temp_array[y+1, x] = INFECTED
                    # self.infect_list[(y+1, x)] = 0
                    tp_list.append((y+1, x))

        #check left
        if x-1 >= 0:
            if self.infection['high'][2] == 1:
                if (y, x-1) not in self.infect_list.keys():
                    temp_array[y, x-1] = INFECTED
                    # self.infect_list[(y, x-1)] = 0
                    tp_list.append((y, x-1))

        #check right
        if x+1 < self.plot_size[0]:
            if self.infection['high'][3] == 1:
                if (y, x+1) not in self.infect_list.keys():
                    temp_array[y, x+1] = INFECTED
                    # self.infect_list[(y, x+1)] = 0
                    tp_list.append((y, x+1))

        #check top left
        if y-1 >= 0 and x-1 >= 0:
            if self.infection['medium'][0] == 1:
                if (y-1, x-1) not in self.infect_list.keys():
                    temp_array[y-1, x-1] = INFECTED
                    # self.infect_list[(y-1, x-1)] = 0
                    tp_list.append((y-1, x-1))

        #check top right
        if y-1 >= 0 and x+1 < self.plot_size[0]:
            if self.infection['medium'][1] == 1:
                if (y-1, x+1) not in self.infect_list.keys():
                    temp_array[y-1, x+1] = INFECTED
                    # self.infect_list[(y-1, x+1)] = 0
                    tp_list.append((y-1, x+1))

        #check bottom left
        if y+1 < self.plot_size[1] and x-1 >= 0:
            if self.infection['medium'][2] == 1:
                if (y+1, x-1) not in self.infect_list.keys():
                    temp_array[y+1, x-1] = INFECTED
                    # self.infect_list[(y+1, x-1)] = 0
                    tp_list.append((y+1, x-1))

        #check bottom right
        if y+1 < self.plot_size[1] and x+1 < self.plot_size[0]:
            if self.infection['medium'][3] == 1:
                if (y+1, x+1) not in self.infect_list.keys():
                    temp_array[y+1, x+1] = INFECTED
                    # self.infect_list[(y+1, x+1)] = 0
                    tp_list.append((y+1, x+1))

        existing_shm.close()
        return tp_list

    def check_neighbour(self):
        temp_list = copy.deepcopy(self.infect_list)
            
        p = Pool(cpu_count())
        x = []
        for i in temp_list.keys():
            x.append((i, self.shm.name))
        results = p.starmap(self.check_neighbour_mp_apply, x)
        # print(results)

        for i in results:
            if isinstance(i, list):
                for j in i:
                    if len(j) == 2:
                        self.infect_list[j] = 0 
            elif len(i) == 2:
                self.infect_list[i] = 0 
        p.close()
        p.join()

    def compute_infection(self, action, plot_state, gs_title):
        self.gs_title = gs_title
        self.plot_state = plot_state
        self.apply_infection_reproductive_stage()
        self.init_shm()
        self.apply_pesticide(action)
        self.spread_probabilities()
        self.infection_tracker()
        self.check_neighbour()
        self.rmv_shm()

        return self.plot_state