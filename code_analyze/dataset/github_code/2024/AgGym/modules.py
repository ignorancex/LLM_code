import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import glob
from PIL import Image
import pdb
import torch
import os
import sys
import random
from modules import threat_modules as tm

EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(1337)

def make_gif():
    frames = [Image.open(image) for image in glob.glob(f"*.png")]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

# class Threat:
# 	def __init__(self, threat_type: str, severity: int, plot_size: tuple[int, int], plot_state):
# 		self.threat_type = threat_type
# 		self.severity = severity
# 		self.plot_size = plot_size

# 	def reset(self):
# 		self.infect_list = []

# 		random_init_x = np.random.randint(0, self.plot_size[1])
# 		random_init_y = np.random.randint(0, self.plot_size[0])
# 		self.infect_list.apapend()

# 	def compute_infection(self):
# 		temp_list = copy.deepcopy(self.infect_list)
# 		for plant_infected in temp_list:
# 			# compute interaction
# 			infection = self.compute_infection()
# 			print(infection)
# 			y, x = plant_infected

# 			#check top
# 			if y-1 >= 0:
# 				if infection['high'][0] == 1:
# 					print('top')
# 					self.plot_state[y-1, x] = INFECTED
# 					self.infect_list.append((x, y-1))

# 			#check bottom
# 			if y+1 < self.plot_size[1]:
# 				if infection['high'][1] == 1:
# 					print('bottom')
# 					self.plot_state[y+1, x] = INFECTED
# 					self.infect_list.append((x, y+1))

# 			#check left
# 			if x-1 >= 0:
# 				if infection['high'][2] == 1:
# 					print('left')
# 					self.plot_state[y, x-1] = INFECTED
# 					self.infect_list.append((x-1, y))

# 			#check right
# 			if x+1 < self.plot_size[0]:
# 				if infection['high'][3] == 1:
# 					print('right')
# 					self.plot_state[y, x+1] = INFECTED
# 					self.infect_list.append((x+1, y))

# 			#check top left
# 			if y-1 >= 0 and x-1 >= 0:
# 				if infection['medium'][0] == 1:
# 					print('top left')
# 					self.plot_state[y-1, x-1] = INFECTED
# 					self.infect_list.append((x-1, y-1))

# 			#check top right
# 			if y-1 >= 0 and x+1 < self.plot_size[0]:
# 				if infection['medium'][1] == 1:
# 					print('top right')
# 					self.plot_state[y-1, x+1] = INFECTED
# 					self.infect_list.append((x+1, y-1))

# 			#check bottom left
# 			if y+1 < self.plot_size[1] and x-1 >= 0:
# 				if infection['medium'][2] == 1:
# 					print('bottom left')
# 					self.plot_state[y+1, x-1] = INFECTED
# 					self.infect_list.append((x-1, y+1))

# 			#check bottom right
# 			if y+1 < self.plot_size[1] and x+1 < self.plot_size[0]:
# 				if infection['medium'][3] == 1:
# 					print('bottom right')
# 					self.plot_state[y+1, x+1] = INFECTED
# 					self.infect_list.append((x+1, y+1))


class AgGym_Env:
	def __init__(self, plot_size: tuple[int, int], num_plants: int):
		self.num_of_plants = num_plants
		self.plot_size = plot_size

		self.plot_state = np.zeros(shape=self.plot_size)
		self.pesticide_actions = {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9}
		# self.prob = [0.5, 0.05, 0.15, 0.3]

		self.t = 0
		self.t_since_last_clean = -1
		# self.key = False

		self.trx = tm.Threat('Bad', 10, self.plot_size, self.pesticide_actions)
		# self.plot_state = self.trx.set_plot_status(self.plot_state, self.t)

		self.growth_stages = {'Planting': 10, 'VE': 5, 'VC': 5, 'V1': 5, 'V2': 5, 'V3': 5, 'V4': 5, 'V5': 3,
							'R1': 3, 'R2': 10, 'R3': 9, 'R4': 9, 'R5': 15, 'R6': 18, 'R7': 9}

		gs_keys = self.growth_stages.keys()
		gs_vals = self.growth_stages.values()
		gs_sum = 0
		self.gs_reverse = {}
		for k, v in zip(gs_keys, gs_vals):
			gs_sum += v
			self.gs_reverse[gs_sum] = k

		for i in range(self.plot_state.shape[0]):
			for j in range(self.plot_state.shape[1]):
				self.plot_state[i][j] = ALIVE

		# fig, ax = plt.subplots(figsize=(9,6))
		# ax = sns.heatmap(self.plot_state, vmin=0, vmax=3)
		for i in self.gs_reverse.keys():
			if i > self.t:
				self.gs_title = self.gs_reverse[i]
				break
		fig, ax = plt.subplots(figsize=(9,6))
		ax = sns.heatmap(self.plot_state, vmin=0, vmax=3)
		plt.title(f"Growth Stage: {self.gs_title}, Day {self.t}")
		plt.savefig("initial_plot_state.png", dpi=300)
		plt.close()

	def reset(self):
		# infect list
		# hardcode for now
		self.t = 0
		self.trx.reset()
		# self.infect_list = {}
		# self.key = False
		# random_init_x = np.random.randint(0, self.plot_size[1])
		# random_init_y = np.random.randint(0, self.plot_size[0])
		# self.infect_list = {(random_init_y, random_init_x):0}
		# for infect in self.infect_list:
		# 	i, j = infect
		# 	self.plot_state[i][j] = INFECTED

		fig, ax = plt.subplots(figsize=(9,6))
		ax = sns.heatmap(self.plot_state, vmin=0, vmax=3)
		for i in self.gs_reverse.keys():
			if i > self.t:
				self.gs_title = self.gs_reverse[i]
				break
		plt.title(f"Growth Stage: {self.gs_title}, Day {self.t}")
		plt.savefig("start_plot_state.png", dpi=300)
		plt.close()

	def step(self, action):
		self.plot_state = self.trx.compute_infection(action, self.plot_state, self.gs_title)
		# if self.key == True and self.t_since_last_clean == -1 and self.gs_title.find('R') != -1:
		# 	self.t_since_last_clean = copy.deepcopy(self.t)

		# if self.gs_title.find('R') != -1 and self.key == False:
		# 	self.key = True
		# 	random_init_x = np.random.randint(0, self.plot_size[1])
		# 	random_init_y = np.random.randint(0, self.plot_size[0])
		# 	self.infect_list = {(random_init_y, random_init_x):0}
		# 	for infect in self.infect_list:
		# 		y, x = infect
		# 		self.plot_state[y, x] = INFECTED

		# if self.t_since_last_clean != -1 and self.t == self.t_since_last_clean + 10:
		# 	print("REINFECTION!")
		# 	random_init_x = np.random.randint(0, self.plot_size[1])
		# 	random_init_y = np.random.randint(0, self.plot_size[0])
		# 	self.infect_list = {(random_init_y, random_init_x):0}
		# 	for infect in self.infect_list:
		# 		y, x = infect
		# 		self.plot_state[y, x] = INFECTED

		# print(f"Infect list before pest: {self.infect_list}")
		# pesticide_chance = self.pesticide_actions[action]
		# temp_list = copy.deepcopy(self.infect_list)
		# pest_chance_list = np.random.choice([0,1], size=len(temp_list), replace=True, p=[1-pesticide_chance, pesticide_chance])
		# for p, i in zip(pest_chance_list, temp_list.keys()):
		# 	if p == 1:
		# 		# pdb.set_trace()
		# 		y, x = i
		# 		print(f"Applying pesticide on {x} {y}")
		# 		self.plot_state[y, x] = ALIVE
		# 		del self.infect_list[i]

		# print(f"Infect list after pest: {self.infect_list}")
		# temp_list = copy.deepcopy(self.infect_list)
		# for plant_infected in temp_list.keys():
		# 	# compute interaction
		# 	infection = self.compute_infection()
		# 	y, x = plant_infected
		# 	print(f"For infected plant {x} {y}")
		# 	print(infection)

		# 	#check top
		# 	if y-1 >= 0:
		# 		if infection['high'][0] == 1:
		# 			if (y-1, x) not in self.infect_list.keys():
		# 				print('top')
		# 				self.plot_state[y-1, x] = INFECTED
		# 				self.infect_list[(y-1, x)] = 0

		# 	#check bottom
		# 	if y+1 < self.plot_size[1]:
		# 		if infection['high'][1] == 1:
		# 			if (y+1, x) not in self.infect_list.keys():
		# 				print('bottom')
		# 				self.plot_state[y+1, x] = INFECTED
		# 				self.infect_list[(y+1, x)] = 0

		# 	#check left
		# 	if x-1 >= 0:
		# 		if infection['high'][2] == 1:
		# 			if (y, x-1) not in self.infect_list.keys():
		# 				print('left')
		# 				self.plot_state[y, x-1] = INFECTED
		# 				self.infect_list[(y, x-1)] = 0

		# 	#check right
		# 	if x+1 < self.plot_size[0]:
		# 		if infection['high'][3] == 1:
		# 			if (y, x+1) not in self.infect_list.keys():
		# 				print('right')
		# 				self.plot_state[y, x+1] = INFECTED
		# 				self.infect_list[(y, x+1)] = 0

		# 	#check top left
		# 	if y-1 >= 0 and x-1 >= 0:
		# 		if infection['medium'][0] == 1:
		# 			if (y-1, x-1) not in self.infect_list.keys():
		# 				print('top left')
		# 				self.plot_state[y-1, x-1] = INFECTED
		# 				self.infect_list[(y-1, x-1)] = 0

		# 	#check top right
		# 	if y-1 >= 0 and x+1 < self.plot_size[0]:
		# 		if infection['medium'][1] == 1:
		# 			if (y-1, x+1) not in self.infect_list.keys():
		# 				print('top right')
		# 				self.plot_state[y-1, x+1] = INFECTED
		# 				self.infect_list[(y-1, x+1)] = 0

		# 	#check bottom left
		# 	if y+1 < self.plot_size[1] and x-1 >= 0:
		# 		if infection['medium'][2] == 1:
		# 			if (y+1, x-1) not in self.infect_list.keys():
		# 				print('bottom left')
		# 				self.plot_state[y+1, x-1] = INFECTED
		# 				self.infect_list[(y+1, x-1)] = 0

		# 	#check bottom right
		# 	if y+1 < self.plot_size[1] and x+1 < self.plot_size[0]:
		# 		if infection['medium'][3] == 1:
		# 			if (y+1, x+1) not in self.infect_list.keys():
		# 				print('bottom right')
		# 				self.plot_state[y+1, x+1] = INFECTED
		# 				self.infect_list[(y+1, x+1)] = 0

		print(self.plot_state)
		fig, ax = plt.subplots(figsize=(9,6))
		ax = sns.heatmap(self.plot_state, vmin=0, vmax=3)
		for i in self.gs_reverse.keys():
			if i > self.t:
				self.gs_title = self.gs_reverse[i]
				break
		plt.title(f"Growth Stage: {self.gs_title}, Day {self.t}")
		plt.savefig(f"step_plot_state_{str(self.t).zfill(6)}.png", dpi=300)
		plt.close()

		self.t += 1


	def compute_infection(self):
		high_infect = np.random.choice([0,1], size=4, replace=True, p=[0.7, 0.3])
		medium_infect = np.random.choice([0,1], size=4, replace=True, p=[0.85, 0.15])
		low_infect = np.random.choice([0,1], size=8, replace=True, p=[0.95, 0.05])

		return {'high': high_infect, 'medium': medium_infect, 'low': low_infect}



env = AgGym_Env((10,10), 9)
env.reset()
# pdb.set_trace()
for i in range(105):
	if i in [46,47]:
		env.step(3)
	# if i == 46:
	# 	env.step(3)
	# elif i == 47:
	# 	env.step(3)
	# elif i == 48:
	# 	env.step(3)
	else:
		env.step(0)
make_gif()
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(0)
# env.step(3)
# env.step(3)
# env.step(3)
# env.step(0)