import numpy as np
from multiagent_envs.multiagent.core import World, Agent, Landmark, Hole, Snack, Obstacle
from multiagent_envs.multiagent.scenario import BaseScenario
import pdb
from vars import pargs
import math
#from train import past
import random
from a2c_ppo_acktr.arguments import get_args


class Scenario(BaseScenario):

	def make_world(self):
		world = World()
		self.args = get_args()
		# set any world properties first
		world.dim_c = 2
		world.time_step = 0
		num_good_agents = pargs.num_agents
		num_adversaries = pargs.num_adv
		num_agents = num_adversaries + num_good_agents
		self.initial_num = num_agents
		num_landmarks = 0
		# add agents
		world.agents = [Agent() for i in range(num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.adversary = True if i < num_adversaries else False
			agent.size = pargs.adv_size if agent.adversary else pargs.prey_size
		# add landmarks
		world.landmarks = [Landmark() for i in range(0)]
		world.safe_holes = [Hole() for i in range(6)]
		world.snacks = [Snack() for i in range(0)]
		world.obs = [Obstacle() for i in range(int(3))]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = True
			landmark.movable = False
			landmark.size = 0.2
			landmark.boundary = False
		# make initial conditions
		self.reset_world(world)
		self.previous_distance_adv = 0
		self.previous_distance_prey = 0
		return world


	def reset_world(self, world):
		#print("reset world")
		# random properties for agents

		world.time_step = 0

		for agents in world.agents:
			agents.state.is_ghost = False
			agents.state.time_since_food = 1
			agents.state.last_locations = []
			agents.state.self_last_locations = []
			agents.state.last_direction = []
			for i in range(50):
				agents.state.last_locations.append([10, 10])

			for i in range(50):
				agents.state.self_last_locations.append([10, 10])
			for i in range(50):
				agents.state.last_direction.append(10)
		for i, agent in enumerate(world.agents):
			agent.color = np.array([0.2, 0.75, 0.55]) if not agent.adversary else np.array([0.5, 0.3, 0.7])
			# random properties for landmarks
		for i, landmark in enumerate(world.landmarks):
			landmark.color = np.array([0.25, 0.25, 0.25])
		# set random initial states


		world.obs_locations = []
		if world.first or True:
			for i, agent in enumerate(world.obs):
				redo = True
				d = 0
				while redo:
					d += 1
					t = 2*3.14159*np.random.random(1)
					t = t[0]
					u = np.random.random()+np.random.random(1)
					u = u[0]
					if u>1:
						r = 2-u
					else:
						r = u
					agent.state.p_pos = np.array([(pargs.region_area - 0.04 * 3 - 0.2)*r*math.cos(t), (pargs.region_area- 0.04 * 3 - 0.2)*r*math.sin(t)])

					redo = False
					for l in world.obs_locations:
						diff = l - agent.state.p_pos
						distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
						if distance < (0.04 * 4 + 0.4) and d < 15:
							redo = True
				world.obs_locations.append(agent.state.p_pos)
				world.first = False

		for i, agent in enumerate(world.agents):
			redo = True
			while redo:
				t = 2*3.14159*np.random.random(1)
				t = t[0]
				u = np.random.random()+np.random.random(1)
				u = u[0]
				if u>1:
					r = 2-u
				else:
					r = u
				agent.state.p_pos = np.array([(pargs.region_area - agent.size)*r*math.cos(t), (pargs.region_area - agent.size)*r*math.sin(t)])
				agent.state.p_vel = np.zeros(world.dim_p)
				agent.state.c = np.zeros(world.dim_c)
				agent.state.direction = 0.0
				agent.state.speed = 1.0
				redo = False
				for l in world.obs_locations:
					diff = l - agent.state.p_pos
					distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
					if distance < (world.obs[0].size + agent.size):
						redo = True


	def benchmark_data(self, agent, world):
		# returns data for benchmarking purposes
		if agent.adversary:
			collisions = 0
			for a in self.good_agents(world):
				if self.is_collision(a, agent):
					collisions += 1
			return collisions
		else:
			return 0

	def get_new_location(self, world):
		redo = True
		while redo:
			t = 2*3.14159*np.random.random(1)
			t = t[0]
			u = np.random.random()+np.random.random(1)
			u = u[0]
			if u>1:
				r = 2-u
			else:
				r = u
			temp_pos = np.array([(pargs.region_area - 0.05)*r*math.cos(t), (pargs.region_area - 0.05)*r*math.sin(t)])
			redo = False
			if len(world.obs_locations) > 0:
				for l in world.obs_locations:
					diff = l - temp_pos
					distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
					if distance < (world.obs[0].size):
						redo = True
		return temp_pos


	def is_collision(self, agent1, agent2):
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = (agent1.size + agent2.size) * 1.5
		#dist_min = 0.2
		return True if dist < dist_min else False

	# return all agents that are not adversaries
	def good_agents(self, world):
		return [agent for agent in world.agents if not agent.adversary]

	# return all adversarial agents
	def adversaries(self, world):
		return [agent for agent in world.agents if agent.adversary]

	def area(self, x1, y1, x2, y2, x3, y3):

		return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
					+ x3 * (y1 - y2)) / 2.0)


	def PointInsideTriangle2(self, pt, tri):
		'''checks if point pt(2) is inside triangle tri(3x2). @Developer'''
		if (-tri[1][1] * tri[2][0] + tri[0][1] * (-tri[1][0] + tri[2][0]) + tri[0][0] * (tri[1][1] - tri[2][1]) + tri[1][0] * tri[2][1]) == 0:
			return 0
		a = 1 / (-tri[1][1] * tri[2][0] + tri[0][1] * (-tri[1][0] + tri[2][0]) + tri[0][0] * (tri[1][1] - tri[2][1]) + tri[1][0] * tri[2][1])
		s = a * (tri[2][0] * tri[0][1] - tri[0][0] * tri[2][1] + (tri[2][1] - tri[0][1]) * pt[0] + (tri[0][0] - tri[2][0]) * pt[1])
		if s < 0:
			return False
		else:
			t = a * (tri[0][0] * tri[1][1] - tri[1][0] * tri[0][1] + (tri[0][1] - tri[1][1]) * pt[0] + (tri[1][0] - tri[0][0]) * pt[1])
		return ((t > 0) and (1 - s - t > 0))


	def reward(self, curr_time, agent, world, agent_specs):
		# Agents are rewarded based on minimum agent distance to each landmark
		main_reward = self.adversary_reward(curr_time, agent, world, agent_specs) if agent.adversary else self.agent_reward(curr_time, agent, world, agent_specs)
		return main_reward

	def agent_reward(self, curr_time, agent, world, agent_specs):
		# Agents are negatively rewarded if caught by adversaries
		#pdb.set_trace()
		close_rew = 0
		vel_rew = 0
		coll_rew = 0
		bound_rew = 0
		view_rew = 0
		done = False

		scale = 1.0
		vel_rew -= 0.2  * scale * agent_specs[3]
		view_rew -= .2 * scale *  agent_specs[2]
		if agent.state.is_ghost:
			return [0, vel_rew, 0.0, view_rew, False]

		agent.state.time_since_food += 1

		shape = True
		adversaries = self.adversaries(world)
		if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
			current_distance = min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in adversaries])
			self.previous_distance_prey = current_distance


		if agent.collide:
			for a in adversaries:
				if self.is_collision(a, agent):
					if not self.is_safe(agent, world):
						agent.state.is_ghost = True
						coll_rew = -5
						done = True

		return [close_rew, vel_rew, coll_rew, view_rew, done]

	def get_alive_agents(self, agents):
		alive = []
		for a in agents:
			if not a.state.is_ghost:
				alive.append(a)
		return alive


	def adversary_reward(self, curr_time, agent, world, agent_specs):
		# Adversaries are rewarded for collisions with agents
		close_rew = 0
		vel_rew = 0
		coll_rew = 0
		bound_rew = 0
		view_rew = 0
		done = False

		#print("adv")


		shape = True
		agents = self.good_agents(world)
		adversaries = self.adversaries(world)
		if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
			alive = self.get_alive_agents(agents)
			if len(alive) == 0:
				current_distance = 0
			else:
				current_distance = min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in alive])
			self.previous_distance_adv = current_distance

		if agent.collide:
			for ag in agents:
				for adv in adversaries:
					if self.is_collision(ag, adv):
						if not ag.state.is_ghost:
							if not self.is_safe(ag, world):
								coll_rew = 5
								done = True

		if world.time_step == 399:
			coll_rew = -1.0
			done = True
		world.time_step += 1


		return [close_rew, vel_rew, coll_rew, view_rew, done]

	def is_safe(self, agent, world):
		safe = False
		return False

	def calc_length(self, ang, area):
		if ang == 0:
			return 1
		return math.sqrt((2*area)/ang)

	def angle_between_vectors(self, vector_1, vector_2):
		angle = np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2))
		return angle

	def is_blocked(self, other, agent, obstacles):
		for obstacle in obstacles:
			agent_pos = agent.state.p_pos
			obstacle_pos = obstacle.state.p_pos
			diff = obstacle_pos - agent_pos
			c = math.sqrt(diff[0]**2 + diff[1]**2)
			d = obstacle.size
			theta = math.acos(d/c)
			a = math.sin(theta)*d
			b = math.cos(theta)*d
			vert = diff*(c-b)/(c)
			horizontal = [diff[1]*a/c, -1* diff[0]*a/c]

			r1 = agent.state.p_pos + vert + horizontal
			r2 = agent.state.p_pos + vert - horizontal

			y1 = (r1 - agent_pos)*100 + agent_pos
			y2 = (r2 - agent_pos)*100 + agent_pos
			dist_to_ag = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
			dist_to_ob = np.sqrt(np.sum(np.square(agent.state.p_pos - r1)))
			if self.PointInsideTriangle2(other.state.p_pos, [agent_pos, y1, y2]) and dist_to_ag > dist_to_ob:
				return True
		return False


	def get_agent_bucket(self, agent_pos, other_pos, agent_dir):

		diff = other_pos - agent_pos
		distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
		theta = self.angle_between_vectors(diff, [math.cos(agent_dir), math.sin(agent_dir)])

		distance = int(8*distance/4.0)
		ang = int(18*(theta + 3.14159)/(3.14159*2))
		num = distance * 18 + ang
		return num


	def observation(self, agent, world, agent_specs):
		is_prey = True
		if agent_specs[2] == 1.0:
			is_prey= False

		# get positions of all entities in this agent's reference frame
		entity_pos = []
		for entity in world.landmarks:
			if not entity.boundary:
				entity_pos.append(entity.state.p_pos - agent.state.p_pos)
		# communication of all other agents
		comm = []
		other_pos = []
		other_vel = []
		for other in world.agents:
			if other is agent: continue

			comm.append(other.state.c)
			dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
			diff = other.state.p_pos - agent.state.p_pos
			distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

			other_ang = -1
			bucket = self.get_agent_bucket(agent.state.p_pos, other.state.p_pos, agent.state.direction)
			future_bucket = 41

			agent.state.last_direction.append(agent.state.direction)
			agent.state.self_last_locations.append(agent.state.p_pos)

			range_radius = math.sqrt(agent_specs[2]/3.1415926)

			if (dist < range_radius) and not self.is_blocked(other, agent, world.obs):
				agent.state.last_locations.append(other.state.p_pos)
				other_ang = (other.state.direction + 3.1415926) % (2 * 3.1415926)
				other_ang = other_ang - 3.1415926
			else:
				agent.state.last_locations.append([10, 10])


			pos = []
			max_back = 1
			for i in range(max_back):
				j = -1 * max_back + i
				if agent.state.last_locations[j][0] == 10:
					pos.append(-1)
					pos.append(-1)
				else:
					diff = agent.state.last_locations[j] - agent.state.p_pos
					distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
					theta = self.angle_between_vectors(diff, [math.cos(agent.state.direction), math.sin(agent.state.direction)])
					pos.append(distance)
					pos.append(theta)

		curr_dir = (agent.state.direction + 3.1415926) % (2 * 3.1415926)
		curr_dir = curr_dir - 3.1415926

		obs_positions = []
		obs_distances = []
		is_touching = []
		for h in world.obs:
			dist = np.sqrt(np.sum(np.square(agent.state.p_pos - h.state.p_pos)))
			safe_spot = self.angle_between_vectors(agent.state.p_pos - h.state.p_pos, [math.cos(agent.state.direction), math.sin(agent.state.direction)])
			diff = h.state.p_pos - agent.state.p_pos
			distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
			obs_distances.append(distance)
			obs_positions.append(safe_spot)

		dir_x = [math.cos(agent.state.direction), math.sin(agent.state.direction)]
		curr_pos = self.angle_between_vectors(agent.state.p_pos, [0, 1])
		global_dist = math.sqrt(agent.state.p_pos[0] ** 2 + agent.state.p_pos[1] ** 2)

		is_hitting = 0
		if (global_dist + .001) > pargs.region_area - agent.size:
			is_hitting = self.angle_between_vectors(agent.state.p_pos, [0, 1])

		just_ate = 0
		if agent.state.time_since_food == 0:
			just_ate = 1

		curr_dir = agent.state.direction % (2 * 3.1415926)
		curr_dir = curr_dir - 3.1415926


		x = np.concatenate([pos] + [[other_ang]] + [obs_positions] + [obs_distances] + [[curr_dir]] + [[curr_pos]] + [[global_dist]] + [[bucket]] + [[future_bucket]])

		return x
