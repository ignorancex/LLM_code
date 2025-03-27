# from modules import threat_modules as tm
import copy
import numpy as np
import os
import sys
import random
import time
from utils import general_utils as gu
from modules import threat_modules as tm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from configobj import ConfigObj
from modules import env_modules

DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

(Path.cwd() / 'tests' / 'plots').mkdir(parents=True, exist_ok=True)

def test_compute_infection_average_time():
	gu.seed_everything(1337)
	size = (10000,10000)
	plot_state = np.full(size, 3.)
	spawn_half_coor = int(size[0]/2)
	plot_state[spawn_half_coor,spawn_half_coor] = INFECTED

	time_list = []
	avg_time_list = []

	threat = tm.Threat('Insect', 10, (size[0],size[0]), {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9})
	for _ in range(10):
		gu.seed_everything(1337)
		threat.reset()
		threat.infect_list = {(spawn_half_coor, spawn_half_coor):0}
		for i in range(100):
			now = time.time()
			plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
			end = time.time()
			time_list.append(end - now)
		# print(f"Average {np.round(np.average(time_list), 6)}, Max: {np.round(max(time_list), 6)}, Min: {np.round(min(time_list), 6)}")
		avg_time_list.append(np.average(time_list))

	print(f"Total Average {np.round(np.average(avg_time_list), 6)}, Max: {np.round(max(avg_time_list), 6)}, Min: {np.round(min(avg_time_list), 6)}")

	assert max(avg_time_list) < 0.03

def test_compute_infection_plot():
	gu.seed_everything(1337)
	size = (10000,10000)
	plot_state = np.full(size, 3.)
	spawn_half_coor = int(size[0]/2)
	plot_state[spawn_half_coor,spawn_half_coor] = INFECTED

	infected_count = []
	time_list = []

	threat = tm.Threat('Insect', 10, (size[0],size[0]), {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9})
	gu.seed_everything(1337)
	threat.reset()
	threat.infect_list = {(spawn_half_coor, spawn_half_coor):0}
	for i in range(200):
		now = time.time()
		plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
		end = time.time()
		time_list.append(end - now)
		infected_count.append(np.count_nonzero(plot_state == 1))
	print(f"Average {np.round(np.average(time_list), 6)}, Max: {np.round(max(time_list), 6)}, Min: {np.round(min(time_list), 6)}")

	ax_dict = plt.figure(num=1, figsize=(9,6), constrained_layout=True, clear=True).subplot_mosaic("""A""")
	sns.lineplot(x=np.arange(len(time_list)), y=time_list, ax=ax_dict["A"], color= 'blue')
	ax_dict["A"].tick_params(axis ='y', labelcolor = 'blue')
	ax_dict["A"].set_ylabel("Average Time Per Computation (s)")
	ax_dict["A"].set_xlabel("Iteration (N)")

	ax2 = ax_dict["A"].twinx()
	sns.lineplot(x=np.arange(len(infected_count)), y=infected_count, ax=ax2, color='red')
	ax2.tick_params(axis ='y', labelcolor = 'red')
	ax2.set_ylabel("Infection Count in List (N)")
	plt.savefig(Path.cwd() / 'tests' / 'plots' / "timing.png", dpi=300)

	assert True == True

def test_compute_infection():
	gu.seed_everything(1337)
	size = (3,3)
	plot_state = np.full(size, 3.)
	spawn_half_coor = int(size[0]/2)

	plot_state[spawn_half_coor,spawn_half_coor] = INFECTED
	# [3,3,3],
	# [3,2,3],
	# [3,3,3]

	threat = tm.Threat('Insect', 10, (size[0],size[1]), {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9})
	threat.reset()
	threat.infect_list = {(spawn_half_coor, spawn_half_coor):0}

	print(plot_state)
	plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
	# [3,3,3],
	# [3,2,3],
	# [2,3,3]
	print(plot_state)

	plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
	# [3,3,3],
	# [3,2,3],
	# [2,3,3]
	print(plot_state)
	plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
	# # [2,2,3],
	# # [2,2,2],
	# # [2,2,3]
	print(plot_state)
	
	assert np.array_equal(plot_state, np.array((
												[2,2,3],
												[2,2,2],
												[2,2,3]
												))
												)

def test_continual_spread_after_death():
	gu.seed_everything(1337)
	size = (3,3)
	plot_state = np.full(size, 3.)
	spawn_half_coor = int(size[0]/2)

	plot_state[spawn_half_coor,spawn_half_coor] = 2
	# [3,3,3],
	# [3,2,3],
	# [3,3,3]

	threat = tm.Threat('Insect', 10, (size[0],size[1]), {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9})
	threat.reset()
	threat.infect_list = {(spawn_half_coor, spawn_half_coor):9}

	print(plot_state)
	plot_state = threat.compute_infection(0, plot_state, 'Planting', 'growthseason')
	# [3,3,3],
	# [3,2,3],
	# [2,3,3]
	print(plot_state)

	assert np.array_equal(plot_state, np.array((
												[3,3,3],
												[3,1,2],
												[3,2,3]
												))
												)

def test_retaining_death():
	gu.seed_everything(1337)
	size = (10,10)
	# spawn_half_coor = int(size[0]/2)
	config = ConfigObj('training_config.ini')
	config['env']['plot_size'] = size
	cwd = Path.cwd()
	config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'retaining_death')
	config['env']['mode'] = 'train'
	config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
	config['env']['growth_stage_title'] = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
	config['env']['growth_stage_days'] = ['3', '10', '9', '9', '15', '18', '9']
	config['env']['sim_mode'] = 'survival'
	config['env']['reward'] = 'expected_yield + exp_yield_pesticide + done_reward'
	config['training']['max_timestep'] = 200

	env = env_modules.insectEnv()
	gu.set_as_attr(env, config)
	env.init()
	# env.plot_state[spawn_half_coor,spawn_half_coor] = INFECTED
	# env.threat.infect_list = {(spawn_half_coor, spawn_half_coor):9}

	key = False
	dead_old = 0

	for ep in range(1):
		obs = env.reset()
		R = 0
		while True:
			obs, reward, done = env.step(1)
			dead_new = np.count_nonzero(env.plot_state == DEAD)
			# print(f"dead_old {dead_old} > dead_new {dead_new}")
			if dead_old > dead_new:
				key = True
			dead_old = dead_new
			R += reward
			reset = env.timestep == env.max_timestep
			if done or reset:
				break
	
	assert key == False

# def test_compute_infection_mp():
# 	gu.seed_everything(1337)
# 	size = (3,3)
# 	plot_state = np.full(size, 3.)
# 	spawn_half_coor = int(size[0]/2)

# 	plot_state[spawn_half_coor,spawn_half_coor] = 2
# 	# [3,3,3],
# 	# [3,2,3],
# 	# [3,3,3]

# 	threat = tm.Threat_mp('Insect', 10, (size[0],size[1]), {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.9})
# 	threat.reset()
# 	threat.infect_list = {(spawn_half_coor, spawn_half_coor):0}

# 	print(plot_state)
# 	plot_state = threat.compute_infection(0, plot_state, 'Planting')
# 	# [3,3,3],
# 	# [3,2,3],
# 	# [2,3,3]
# 	print(plot_state)

# 	plot_state = threat.compute_infection(0, plot_state, 'Planting')
# 	# [3,3,3],
# 	# [3,2,3],
# 	# [2,3,3]
# 	print(plot_state)
# 	plot_state = threat.compute_infection(0, plot_state, 'Planting')
# 	# # [2,2,3],
# 	# # [2,2,2],
# 	# # [2,2,3]
# 	print(plot_state)
	
# 	assert True