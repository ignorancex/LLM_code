from turtle import fd
import numpy as np
from utils import env_utils
from modules import threat_modules_pestval as tm
from utils import general_utils as gu
import gym
import logging
from PIL import Image
from pathlib import Path
from datetime import datetime
from modules.agents_module.reward import reward_accumulator
from modules.agents_module.action import action_delegator
from modules.agents_module.state import state_accumulator
from modules.agents_module.done import done_delegator
import time

import cv2
import shapefile
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import seaborn as sns
import matplotlib as mpl
import pandas as pd

EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0
MRES = 4.0
import pdb 
class cartesian_grid:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init(self, size=None, multigrid=None, multi_coord=None, measurements=None):

        if measurements == None: # first level setup
            self.baseL = int(self.base_total_length)
            self.baseW = int(self.base_total_width)
            self.plotL = int(self.base_plot_length)
            self.plotW = int(self.base_plot_width)

            self.baseL_ml = int(self.multilvl_total_length)
            self.baseW_ml = int(self.multilvl_total_width)
            self.plotL_ml = int(self.multilvl_plot_length)
            self.plotW_ml = int(self.multilvl_plot_width)
            
            # self.w = int(self.w)
            # self.h = int(self.h)
            # self.ow = int(self.ow)
            # self.oh = int(self.oh)
        else:
            self.shpfile = None
            self.baseL, self.baseW, self.plotL, self.plotW = measurements
            
        self.baseL_div = int(self.baseL / self.plotL)
        self.baseW_div = int(self.baseW / self.plotW)
        self.coords_x = []
        self.coords_y = []
        self.multi_res = []
    
        self.multi_lvl = False
        
        if multi_coord != None:
            self.n, self.s, self.e, self.w = multi_coord
            
        if (self.shpfile == None) and not (isinstance(multigrid, np.ndarray)):
            # If no shapefile provided
            self.size = self.plot_size
            self.grid = np.zeros(self.size)
        elif isinstance(multigrid, np.ndarray):
            # This level is multilvl
            self.grid = multigrid
        else:
            plot_L = self.baseL / self.plotL_ml
            plot_W = self.baseW / self.plotW_ml
            # print(f"plot_L {plot_L} = self.baseL {self.baseL} / self.plotL_ml {self.plotL_ml}")
            # print(f"plot_W {plot_W} = self.baseW {self.baseW} / self.plotW_ml {self.plotW_ml}")
            self.size = (self.baseL, self.baseW)
            print(f"self.size {self.size} = (plot_L {plot_L}, plot_W {plot_W})")
            
            sf = shapefile.Reader(self.shpfile)
            shapes = sf.shapes()[0]

            npoints = len(shapes.points) # total points
            nparts = len(shapes.parts) # total parts
            print(nparts)
            print(npoints)
            if nparts == 1:
                fig, ax_dict = plt.subplot_mosaic("A", figsize=(20,20), constrained_layout=True, clear=True)
                canvas = FigureCanvas(fig)
                ax_dict["A"].axis('off')

                x_lon = np.zeros((len(shapes.points),1))
                y_lat = np.zeros((len(shapes.points),1))

                for ip in range(len(shapes.points)):
                    x_lon[ip] = shapes.points[ip][0]
                    y_lat[ip] = shapes.points[ip][1]
                ax_dict["A"].plot(x_lon,y_lat,'k')
                
                canvas.draw()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                image_from_plot = np.flip(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)),0)
   
                img_gray = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2GRAY)
                # Convert ndarray to an image object
                

                west = x_lon.min()
                east = x_lon.max()
                north = y_lat.max()
                south = y_lat.min()
                
                print(f'Lon and Lat are {x_lon}, {y_lat}')
                #middle = (int(south + (north - south)/2), int(west + (east - west)/2))
                middle = ((south + (north - south)/2), (west + (east - west)/2))

                canvas_height, canvas_width = img_gray.shape
                print(f'canvas height - {canvas_height}')
                print(f'middle - {middle[0]}')
                #print(( (middle[0] - south)/(north - south)))
                
                # scaled_middle_height = int(( (middle[0] - south)/(north - south) ) * (canvas_height) + 0)
                # scaled_middle_width = int(( (middle[1] - west)/(east - west) ) * (canvas_width) + 0)
                scaled_middle_height = int(( (middle[0] - south)/(north - south) ) * (canvas_height) + 0)
                scaled_middle_width = int(( (middle[1] - west)/(east - west) ) * (canvas_width) + 0)
                print(f'scaled_middle_height - {scaled_middle_height}')
                # print(scaled_middle_width)

                filled_image = flood_fill(image=img_gray, seed_point=(scaled_middle_height, scaled_middle_width), new_value=0)
                plt.close()
                
                self.grid = cv2.resize(filled_image, dsize=(int(self.size[0]), int(self.size[1])), interpolation=cv2.INTER_NEAREST)

            else:
                idx = 0
                for ip in range(nparts):
                    fig, ax_dict = plt.subplot_mosaic("A", figsize=(20,20), constrained_layout=True, clear=True)
                    canvas = FigureCanvas(fig)
                    ax_dict["A"].axis('off')

                    i0=shapes.parts[ip]
                    if ip < nparts-1:
                        i1 = shapes.parts[ip+1]-1
                    else:
                        i1 = npoints

                    if idx != 0: # if not base canvas, plot base canvas to get a similar sized canvas with the base
                        ax_dict["A"].plot(grid_x_lon, grid_y_lat, 'k')
                    seg=shapes.points[i0:i1+1]
                    x_lon = np.zeros((len(seg),1))
                    y_lat = np.zeros((len(seg),1))
                    for ip in range(len(seg)):
                        x_lon[ip] = seg[ip][0]
                        y_lat[ip] = seg[ip][1]
                    ax_dict["A"].plot(x_lon,y_lat,'k')
                    if idx != 0: # if not base canvas, since any other parts are smaller than base canvas, draw over the 
                                # lines used as a placeholder for base canvas size
                        ax_dict["A"].plot(grid_x_lon, grid_y_lat, 'w', linewidth=5)

                    canvas.draw()
                    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                    image_from_plot = np.flip(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)),0)
                    img_gray = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2GRAY)
                    
                    locate_y, locate_x = np.where(img_gray==0)
                    west = locate_x.min()
                    east = locate_x.max()
                    north = locate_y.max()
                    south = locate_y.min()
                    middle = (int(south + (north - south)/2), int(west + (east - west)/2))
                    canvas_height, canvas_width = img_gray.shape
                    scaled_middle_height = int(( (middle[0] - south)/(north - south) ) * (canvas_height) + 0)
                    scaled_middle_width = int(( (middle[1] - west)/(east - west) ) * (canvas_width) + 0)
                    filled_image = flood_fill(image=img_gray, seed_point=(scaled_middle_height, scaled_middle_width), new_value=0)
                    plt.close()

                    if idx == 0: #base canvas
                        self.grid = cv2.resize(filled_image, dsize=(int(self.size[0]), int(self.size[1])), interpolation=cv2.INTER_NEAREST)
                        grid_x_lon = x_lon
                        grid_y_lat = y_lat

                    else:
                        multi_grid = cv2.resize(filled_image, dsize=(int(self.size[0]), int(self.size[1])), interpolation=cv2.INTER_NEAREST)
                        locate_y, locate_x = np.where(multi_grid==0)
                        west = locate_x.min()
                        east = locate_x.max()
                        north = locate_y.max()
                        south = locate_y.min()
                        self.multigrid_add((self.baseL_ml, self.baseW_ml), multi_grid, north, south, east, west, self.baseL_ml, self.baseW_ml, self.plotL_ml, self.plotW_ml)

                    idx += 1
                    
            self.allocate_coords()
            self.assign_multi_grid()
            self.remap_grid(ALIVE)

    def allocate_coords(self, shape=None):
        if shape == None:
            # shape = self.size
            # self.L_div = shape[0] / self.baseL_div
            # self.W_div = shape[1] / self.baseW_div
            # print(f"self.L_div {self.L_div} = shape[0] {shape[0]} / self.baseL_div {self.baseL_div}")
            # print(f"self.W_div {self.W_div} = shape[1] {shape[1]} / self.baseW_div {self.baseW_div}")
            self.L_div = self.plotL
            self.W_div = self.plotW
            print(f"self.L_div: {self.L_div}, self.W_div: {self.W_div}")

        else:
            # ratioL = self.baseL / shape[0] 
            # ratioW = self.baseW / shape[1]
            # self.L_div = shape[2] * ratioL
            # self.W_div = shape[3] * ratioW
            # print(f"self.L_div {self.L_div} = shape[2] {shape[2]} * ratioL {ratioL}")
            # print(f"self.W_div {self.W_div} = shape[3] {shape[3]} * ratioW {ratioW}")
            self.L_div = shape[0]
            self.W_div = shape[1]
            print(f"self.L_div: {self.L_div}, self.W_div: {self.W_div}")

        locate_y, locate_x = np.where(self.grid==0)

        for i,j in zip(locate_x, locate_y):
            if i % self.L_div == 0 and j % self.W_div == 0:
                self.coords_x.append(i)
                self.coords_y.append(j)
                
        assert len(self.coords_x) > 0, f"len(self.coords_x): {len(self.coords_x)} < 0"
        assert len(self.coords_y) > 0, f"len(self.coords_y): {len(self.coords_y)} < 0"
        
        if self.multi_lvl:
            for lvl in self.multi_res:
                lvl.allocate_coords((self.plotL_ml, self.plotW_ml))
                
    def assign_multi_grid(self):
        temp_x = []
        temp_y = []
        
        temp_coords_x = list(self.coords_x)
        temp_coords_y = list(self.coords_y)
        remove_index_list = []
        
        if self.multi_lvl:
            for lvl in self.multi_res:
                assert len(self.coords_x) == len(self.coords_y)
                for r in range(len(self.coords_x)):
                    for li, lj in zip(lvl.coords_x, lvl.coords_y):
                        bi = self.coords_x[r]
                        bj = self.coords_y[r]
                        if bi == li and bj == lj:
                            temp_x.append(bi)
                            temp_y.append(bj)
                            remove_index_list.append(r)
                        else:
                            pass

        for i in remove_index_list[::-1]: # reverse index list to pop, so that list dynamically shrink
                                          # from the back, which does not affect popping operation
            temp_coords_x.pop(i)
            temp_coords_y.pop(i)

        self.coords_x = temp_coords_x
        self.coords_y = temp_coords_y

        for i, j in zip(temp_x, temp_y):
            self.grid[j,i] = MRES
            
    def remap_grid(self, val):
        temp_grid = np.zeros(self.grid.shape)
        
        for idx, jdx in zip(self.coords_x, self.coords_y):
            temp_grid[jdx, idx] = val
            
        if self.multi_lvl:
            for lvl in self.multi_res:
                lvl.remap_grid(val)
                lvl_y, lvl_x = np.where(lvl.grid == val)
                for idx, jdx in zip(lvl_x, lvl_y):
                    temp_grid[jdx, idx] = val

        return temp_grid
        
    def multigrid_add(self, resolution, grid, n, s, e, w, bL, bW, pL, pW):
        env = cartesian_grid()
        env.init(size=resolution, multigrid=grid, multi_coord=(n, s, e, w), measurements=(bL, bW, pL, pW))
        self.multi_res.append(env)
        self.multi_lvl = True

    def render(self, ax, val):
        plot_state_cmap = sns.color_palette("rocket", as_cmap=True)
        norm = mpl.colors.Normalize(vmin=0, vmax=3)
        ax[val].scatter(self.coords_x, self.coords_y, color='w', s=10)
        for x,y in zip(self.coords_x, self.coords_y):
            if self.grid[y,x] == ALIVE:
                rectangle = plt.Rectangle((x-(self.plotL/10)*4, y-(self.plotW/10)*4), (self.plotL/10)*9, (self.plotW/10)*9, fc=plot_state_cmap(norm(3)))
                ax[val].add_patch(rectangle)
            elif self.grid[y,x] == INFECTED:
                rectangle = plt.Rectangle((x-(self.plotL/10)*4, y-(self.plotW/10)*4), (self.plotL/10)*9, (self.plotW/10)*9, fc=plot_state_cmap(norm(2)))
                ax[val].add_patch(rectangle)
            elif self.grid[y,x] == DEAD:
                rectangle = plt.Rectangle((x-(self.plotL/10)*4, y-(self.plotW/10)*4), (self.plotL/10)*9, (self.plotW/10)*9, fc=plot_state_cmap(norm(1)))
                ax[val].add_patch(rectangle)
            else:
                continue

        if self.multi_lvl:
            for lvl in self.multi_res:
                lvl.render(ax[val])
            return ax
        else:
            return ax
    def clean_and_convert(self, value):
        cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
        return float(cleaned_value)   
    def reshape_to_max_square(self, vec):
    # Reshape a vector to the largest possible square matrix, padding with zeros
    # Find the dimension of the target matrix
        dim = int(np.ceil(np.sqrt(len(vec))))
    # Calculate how many zeros need to be added
        padding = (dim * dim) - len(vec)
    
    # Add the zeros to the end of the vector
        padded_vec = np.pad(vec, (0, padding))
    
    # Reshape the vector to the matrix form
        matrix = padded_vec.reshape(dim, dim)
    
        return matrix 
    def rl_init(self):
        self.growth_stage_days = gu.str_to_type(self.growth_stage_days, 'int')
        self.action_dim = gu.str_to_type(self.action_dim, 'float')
        self.max_timestep = int(self.max_timestep)
        self.severity = float(self.severity)
        
        self.grid = self.remap_grid(ALIVE)
        self.healthy_grid = self.remap_grid(ALIVE)
        self.terminal_grid = self.remap_grid(DEAD)
        self.growth_stages = {}
        for i,j in zip(self.growth_stage_title, self.growth_stage_days):
            self.growth_stages[i] = j
        
        self.action_list = []
        self.infect_counts = []
        self.dead_counts = []
        self.dead_counts2 = []
        
        self.num_of_plots_per_acre = int(43560 / (self.plotL * self.plotW))
        self.unit_potential_yield = int(self.simulated_potential_yield) / self.num_of_plots_per_acre
        self.price_per_bushel = int(self.price_per_bushel)
        print(self.price_per_acre) 
        self.unit_pesticide_per_acre = [self.clean_and_convert(value)/ self.num_of_plots_per_acre for value in self.price_per_acre] 
        print(self.unit_pesticide_per_acre)
        self.timestep = 0
        self.episode = -1
        self.action = 0
        self.reward_list = []
        self.state_space = len(self.coords_x)
        print(f"state space {self.state_space}")
        print(f"Grid size: {self.grid.shape}")

        reward_accumulator.init(self)
        action_delegator.init(self)
        state_accumulator.init(self)
        done_delegator.init(self)

        env_utils.set_growth_stage_dict(self)
        env_utils.retrieve_growth_stage(self)
        if self.mode == "eval":
            env_utils.plot_field(self, "initial_grid")

        logging.info("---------- Environment Parameters ----------")
        logging.info(f"Action Space: {self.action_space.n}")
        logging.info(f"State Space: {self.grid.shape}, {self.state_space}")
        logging.info(f"Growth Stages: {self.growth_stages}")
        logging.info("---------- Environment log end ----------")


        apsim_df = pd.read_csv(self.cropmodel_file)
        gridmet_df = pd.read_csv(self.weather_file)
        gridmet_df = gridmet_df.set_index('datetime')
        def strip(str):
            x,y,z = str.split('-')
            return x
        test2 = apsim_df['Date'].array
        iteryears = np.unique(np.array(list(map(strip, test2))))
        
        self.stagename_list = []
        self.bushels_list = []
        self.precipitation_list = []
        self.temp_list = []
        
        self.list_length = 0

        # for i in range(len(iteryears)-1):
        #     begining = iteryears[i]
        #     end = iteryears[i+1]
        #     test1 = apsim_df.loc[(apsim_df['Soybean.AboveGround.Wt']!=0) & ( (apsim_df['Clock.Today'] >= f'{begining}-01-01 00:00:00') & (apsim_df['Clock.Today'] < f'{end}-01-01 00:00:00') )]
        #     self.stagename_list.append(test1['Soybean.Phenology.CurrentStageName'].array)
        #     self.bushels_list.append(test1['AdjBushels'].array[-1])
        #     start_season = test1['Date'].array[0]
        #     end_season = test1['Date'].array[-1]
        #     self.precipitation_list.append(gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['pr'].array)
        #     temp_min = gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['tmmn'].to_numpy()
        #     temp_max = gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['tmmx'].to_numpy()
        #     self.temp_list.append(((temp_max+temp_min)/2)-273.15)
        #     self.list_length += 1

        begining = int(self.year)
        end = int(self.year) + 1
        
        test1 = apsim_df.loc[(apsim_df['Soybean.AboveGround.Wt']!=0) & ( (apsim_df['Clock.Today'] >= f'{begining}-01-01 00:00:00') & (apsim_df['Clock.Today'] < f'{end}-01-01 00:00:00') )]
        self.stagename_list.append(test1['Soybean.Phenology.CurrentStageName'].array)
        self.bushels_list.append(test1['AdjBushels'].array[-1])
        start_season = test1['Date'].array[0]
        end_season = test1['Date'].array[-1]
        self.precipitation_list.append(gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['pr'].array)
        temp_min = gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['tmmn'].to_numpy()
        temp_max = gridmet_df[f'{start_season} 00:00:00': f'{end_season} 00:00:00']['tmmx'].to_numpy()
        self.temp_list.append(((temp_max+temp_min)/2)-273.15)
        self.list_length += 1

        self.threat = eval(f"tm.{self.threat_name}({self.plotL*2}, {self.plotW*2}, {self.plotL*4}, {self.plotW*4}, self.grid, self.coords_x, self.coords_y, self.pesticide_actions)")

    def reset(self):
        self.grid = self.remap_grid(ALIVE)
        self.timestep = 0
        self.episode += 1
        self.reward_list = []
        self.action_list = []
        self.action = 0
        self.threat.reset()
        self.index = 0
        
        if self.sim_from_data == 'True':
            self.index = np.random.choice(range(self.list_length))
            self.unit_potential_yield = self.bushels_list[self.index] / self.num_of_plots_per_acre
            # self.unit_potential_yield = 64 / self.num_of_plots_per_acre
            time_day = []
            time_count = 1
            for i in self.stagename_list[self.index][1:]:
                if isinstance(i, str):
                    time_day.append(time_count)
                    time_count = 1
                else:
                    time_count += 1
            time_day.append(1)
            idx = 0
            self.growth_stages = {}
            for stg_name in self.stagename_list[self.index].unique():
                if isinstance(stg_name, str):
                    self.growth_stages[stg_name] = time_day[idx]
                    idx += 1
                else:
                    pass
        
        #TEMP
        self.key = False
        self.infect_counts = []
        self.dead_counts = []
        self.Degraded_list=[]
        self.dead_counts2 = []
        self.infectdeg_list = []

        env_utils.set_growth_stage_dict(self)
        env_utils.retrieve_growth_stage(self)
        # if self.mode == "eval" or (self.mode == "train" and self.timestep % 1000 == 0 and self.plot_progression == 'True'):
        #     env_utils.plot_grid(self, "start_grid")
        return state_accumulator.main(self)
        
    def step(self, action):

        self.action = action
        start = time.time()
        self.grid, Degraded_list, infectdeg_list = self.threat.compute_infection(action, self.grid, self.gs_title, self.timestep, self.sim_mode, self.withpest_val, self.severity)
        end = time.time()
        logging.debug(f"Timestep: {self.timestep}, spread_computation: {end - start}")
        # if self.sim_from_data == 'True' and self.threat.key == True and self.key == False:
        #     temp = self.temp_list[self.index][self.timestep]
        #     prec = self.precipitation_list[self.index][self.timestep]
        #     odds = 66.571 - 0.74 * prec - 2.594 * temp + 0.026 * prec * temp
        #     self.severity = np.exp(odds) / (1 + np.exp(odds))
        #     self.key = True
        if self.sim_from_data == 'True':

            if self.timestep < len(self.temp_list[0]):
                
                temp = self.temp_list[0][self.timestep]
                prec = self.precipitation_list[0][self.timestep]
            else:
                temp = self.temp_list[0][-1]
                prec = self.precipitation_list[0][-1]
            odds = 66.571 - 0.74 * prec - 2.594 * temp + 0.026 * prec * temp
            self.severity = np.exp(odds) / (1 + np.exp(odds))
            self.key = True
            self.temp=temp
            self.prec=prec
            # self.severity =0.06
        # Done condition
        done = done_delegator.main(self)

        # Reward
        reward = reward_accumulator.main(self)
        infect_count = np.count_nonzero(self.grid == INFECTED)
        dead_count = np.count_nonzero(self.grid == DEAD)
        self.reward_list.append(reward)
        self.action_list.append(action)
        self.infect_counts.append(infect_count)
        self.dead_counts.append(dead_count)
        # self.Degraded_list.append(np.sum(np.array(Degraded_list)))
        if self.timestep == 0:
            self.dead_counts2.append(dead_count)

        elif self.dead_counts2[-1]!=dead_count:
            self.dead_counts2.append(dead_count)
            # self.Degraded_list.append(np.sum(np.array(Degraded_list)))
            if len(Degraded_list):
                for i in range(len(Degraded_list)):
                    self.Degraded_list.append(Degraded_list[i])
        #self.Degraded_list=Degraded_list

        if len(infectdeg_list):
            self.infectdeg_list.append(np.sum(np.array(infectdeg_list)))
        env_utils.retrieve_growth_stage(self)

        if self.mode == "eval" or (self.mode == "train" and self.episode % int(self.max_episode) == 0 and self.plot_progression == 'True'):
             # or (self.mode == "train" and self.episode % 1000 == 0)
            env_utils.plot_field(self, f"step_plot_state_{str(self.timestep).zfill(6)}")
        self.timestep += 1

        return state_accumulator.main(self), reward, done

# class insectEnv(baseEnv):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

# class diseaseEnv(baseEnv):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)