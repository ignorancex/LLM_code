import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent_envs.multiagent.multi_discrete import MultiDiscrete
import pdb
import math
from vars import pargs
import pdb
from a2c_ppo_acktr.arguments import get_args

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, world, reset_callback=None, reward_callback=None,
				 observation_callback=None, info_callback=None,
				 done_callback=None, shared_viewer=True):

		self.world = world
		self.args = get_args()
		self.world.ghost_agents = []
		self.agents = self.world.policy_agents
		# set required vectorized gym env property
		self.n = len(world.policy_agents)
		# scenario callbacks
		self.reset_callback = reset_callback
		self.reward_callback = reward_callback
		self.observation_callback = observation_callback
		self.info_callback = info_callback
		self.done_callback = done_callback
		# environment parameters
		self.discrete_action_space = True
		# if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
		self.discrete_action_input = False
		# if true, even the action is continuous, action will be performed discretely
		self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
		# if true, every agent has the same reward
		self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
		self.time = 0

		# configure spaces
		self.action_space = []
		self.observation_space = []
		for agent in self.agents:
			total_action_space = []
			# physical action space
			if self.discrete_action_space:
				u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
			else:
				u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
			if agent.movable:
				total_action_space.append(u_action_space)
			# communication action space
			if self.discrete_action_space:
				c_action_space = spaces.Discrete(world.dim_c)
			else:
				c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
			if not agent.silent:
				total_action_space.append(c_action_space)
			# total action space
			if len(total_action_space) > 1:
				# all action spaces are discrete, so simplify to MultiDiscrete action space
				if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
					act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
				else:
					act_space = spaces.Tuple(total_action_space)
				self.action_space.append(act_space)
			else:
				self.action_space.append(total_action_space[0])
			# observation space
			#print(self.action_space)
			#exit()
			x = observation_callback(agent, self.world, [0.5, 0.5, 0.5, 0.5])
			obs_dim = len(x)
			self.observation_space = np.array(x)
			agent.action.c = np.zeros(self.world.dim_c)

		# rendering
		self.shared_viewer = shared_viewer
		if self.shared_viewer:
			self.viewers = [None]
		else:
			self.viewers = [None] * self.n
		self._reset_render()

	def angle_between_vectors(self, vector_1, vector_2):
		angle = np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2))
		return angle


	def step(self, action_n):
		curr_time = 0

		if self.args.vision == 'short':
			visual_range = 0.3
		elif self.args.vision == 'medium':
			visual_range = 1.0
		elif self.args.vision == 'long':
			visual_range = 50.0

		if self.args.speed == 'veryslow':
			pred_speed = 0.5
		elif self.args.speed == 'slow':
			pred_speed = 0.55
		elif self.args.speed == 'average':
			pred_speed = 0.60
		elif self.args.speed == 'fast':
			pred_speed = 0.65
		elif self.args.speed == 'veryfast':
			pred_speed = 0.70

		agent_specs = [[1.5,1.5, visual_range, pred_speed],[1.5,1.5,1.0,0.5]]

		obs_n = []
		true_angle = []
		true_relative = []
		true_pos = []
		pred_dir = []

		close = []
		vel = []
		coll = []
		bound = []
		reward_n = []

		done_n = []
		info_n = {'n': []}
		self.agents = self.world.policy_agents
		# set action for each agent
		for i, agent in enumerate(self.agents):
			agent.action.u = action_n[i]
			agent.max_speed = agent_specs[i][3]
		# advance world state
		self.world.step()
		# record observation for each agent
		i = 0
		#print(self.agents)
		for agent in self.agents:
			ob_curr = self._get_obs(agent, agent_specs[i])
			obs_n.append(ob_curr)

			r = self._get_reward(curr_time, agent, agent_specs[i])

			close.append(np.array(r[0]))
			vel.append(np.array(r[1]))
			coll.append(np.array(r[2]))
			bound.append(np.array(r[3]))

			#reward_n.append(self._get_reward(agent))
			done_n.append(r[4])

			info_n['n'].append(self._get_info(agent))
			i += 1

		# all agents get total reward in cooperative case
		reward = np.sum(reward_n)
		if self.shared_reward:
			reward_n = [reward] * self.n

		return obs_n, coll, any(done_n), info_n

	def reset(self):
		agent_specs = [[1.5,1.5,.0001,1.5],[1.5,1.5,1.0,1.5]]
		# reset world
		self.reset_callback(self.world)
		# reset renderer
		self._reset_render()
		# record observations for each agent
		obs_n = []
		self.agents = self.world.policy_agents
		i = 0
		for agent in self.agents:
			obs_n.append(self._get_obs(agent, agent_specs[i]))
			i += 1
		return obs_n

	# get info used for benchmarking
	def _get_info(self, agent):
		if self.info_callback is None:
			return {}
		return self.info_callback(agent, self.world)

	# get observation for a particular agent
	def _get_obs(self, agent, agent_specs):
		if self.observation_callback is None:
			return np.zeros(0)
		return self.observation_callback(agent, self.world, agent_specs)

	# get dones for a particular agent
	# unused right now -- agents are allowed to go beyond the viewing screen
	def _get_done(self, agent):
		if self.done_callback is None:
			return False
		return self.done_callback(agent, self.world)

	# get reward for a particular agent
	def _get_reward(self, curr_time, agent, agent_specs):
		if self.reward_callback is None:
			return 0.0
		return self.reward_callback(curr_time, agent, self.world, agent_specs)

	# reset rendering assets
	def _reset_render(self):
		self.render_geoms = None
		self.render_geoms_xform = None

	def calc_length(self, ang, area):
		if ang == 0:
			return 1
		return math.sqrt((2*area)/ang)


	def make_view(self, ang, ang_between, side_len):

		points = []
		points.append([0, 0])
		ang_span = (ang * 2.0)/10.0

		for i in range(11):
			p = [math.cos(-ang + ang_between + ang_span*i) * side_len, math.sin(-ang + ang_between + ang_span*i ) * side_len]
			points.append(p)

		return points

	def get_leg_pos(self, time):
		angle = time%8
		print(time)
		step_angle = 0.1 * angle
		return [math.acos(step_angle) * 0.04, math.asin(step_angle) * 0.04]


	def is_blocked(self, agent, obstacle, num):
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
		x = [r1, r2]

		return x[num]



	# render environment
	def render(self, mode='human'):

		time_angle = [1.3, 1.3, 1.3, 1.3, 1.3]
		seeing = [True, True]

		if self.args.vision == 'short':
			visual_range = 0.3
		elif self.args.vision == 'medium':
			visual_range = 1.0
		elif self.args.vision == 'long':
			visual_range = 0.00001

		agent_specs = [[1.5,1.50,visual_range, 0.05],[1.5,1.5,1.0,0.05]]

		if mode == 'human':
			alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
			message = ''
			for agent in self.world.agents:
				comm = []
				for other in self.world.agents:
					if other is agent: continue
					if np.all(other.state.c == 0):
						word = '_'
					else:
						word = alphabet[np.argmax(other.state.c)]
					message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
			#print(message)

		for i in range(len(self.viewers)):
			# create viewers (if necessary)
			if self.viewers[i] is None:
				# import rendering only if we need it (and don't import for headless machines)
				#from gym.envs.classic_control import rendering
				from multiagent_envs.multiagent import rendering
				self.viewers[i] = rendering.Viewer(1000,1000)

		# create rendering geometry
		if self.render_geoms is None:
			# import rendering only if we need it (and don't import for headless machines)
			#from gym.envs.classic_control import rendering
			from multiagent_envs.multiagent import rendering
			self.render_geoms_hole = []
			self.render_geoms_hole_xform = []


			jj = 0
			for h in self.world.safe_holes:
				geom_h = rendering.make_circle(.03, res=60)
				geom_h.set_color(1, 0, 0, alpha=0.0)
				xform_h = rendering.Transform()
				w = rendering.LineWidth(1)
				geom_h.add_attr(xform_h)
				geom_h.add_attr(w)
				self.render_geoms_hole.append(geom_h)
				self.render_geoms_hole_xform.append(xform_h)
				jj += 1


			self.render_geoms_obs = []
			self.render_geoms_obs_xform = []
			for h in self.world.obs:
				geom_h = rendering.make_circle(h.size, res=60)
				geom_h.set_color(.5, .5, .5, alpha=1)
				xform_h = rendering.Transform()
				w = rendering.LineWidth(1)
				geom_h.add_attr(xform_h)
				geom_h.add_attr(w)
				self.render_geoms_obs.append(geom_h)
				self.render_geoms_obs_xform.append(xform_h)

			self.render_geoms_snacks = []
			self.render_geoms_snacks_xform = []
			for s in self.world.snacks:
				geom_s = rendering.make_circle(s.size, res=60)
			   # geom_h.set_linewidth(2)
				geom_s.set_color(0, .5, .5, alpha=0.5)
				xform_s = rendering.Transform()
				geom_s.add_attr(xform_s)
				self.render_geoms_snacks.append(geom_s)
				self.render_geoms_snacks_xform.append(xform_s)


			self.render_geoms = []
			self.render_geoms_xform = []
			self.render_geoms_range = []
			self.render_geoms_xform_range = []
			self.render_geoms_head = []
			self.render_geoms_xform_head = []
			self.render_geoms_leg_1 = []
			self.render_geoms_xform_leg_1 = []
			self.render_geoms_leg_2 = []
			self.render_geoms_xform_leg_2 = []
			self.render_geoms_leg_3 = []
			self.render_geoms_xform_leg_3 = []
			self.render_geoms_leg_4 = []
			self.render_geoms_xform_leg_4 = []
			i = 0
			for entity in self.world.entities:
				geom = rendering.make_circle(entity.size)
				geom_head = rendering.make_circle(entity.size * 0.5)
				geom_leg_1 = rendering.PolyLine([[0, 0], [.02*.4*agent_specs[i][3], 0.06*.4*agent_specs[i][3]]], True, width=max(agent_specs[i][3]*6, 1))
				geom_leg_2 = rendering.PolyLine([[0, 0], [0.02*.4*agent_specs[i][3], -0.06*.4*agent_specs[i][3]]], True, width=max(agent_specs[i][3]*6, 1))

				geom_leg_3 = rendering.PolyLine([[0, 0], [.02*.4*agent_specs[i][3], 0.06*.4*agent_specs[i][3]]], True, width=max(agent_specs[i][3]*6, 1))
				geom_leg_4 = rendering.PolyLine([[0, 0], [0.02*.4*agent_specs[i][3], -0.06*.4*agent_specs[i][3]]], True, width=max(agent_specs[i][3]*6, 1))

				range_radius = math.sqrt(agent_specs[i][2]/3.1415926)

				#view_1 = self.make_view(ang, ang_between, side_len)
				#geom_range_1 = rendering.make_polygon(view_1, filled=True)
				geom_range_1 = rendering.make_circle(range_radius, res=60)

				# view_2 = self.make_view(ang, -ang_between, side_len)
				# geom_range_2 = rendering.make_polygon(view_2, filled=True)

				geom_range_2 = rendering.make_circle(range_radius, res=60)


				geom_range_1.set_color(1, 0.8, 0, alpha=0.1)
				geom_range_2.set_color(1, 0.8, 0, alpha=0.1)

				xform = rendering.Transform()
				xform_range = rendering.Transform()
				xform_head = rendering.Transform()
				xform_leg_1 = rendering.Transform()
				xform_leg_2 = rendering.Transform()
				xform_leg_3 = rendering.Transform()
				xform_leg_4 = rendering.Transform()
				if 'agent' in entity.name:
					geom.set_color(*entity.color, alpha=0.5)
					geom_head.set_color(*entity.color, alpha=0.5)
					geom_leg_1.set_color(*entity.color, alpha=0.5)
					geom_leg_2.set_color(*entity.color, alpha=0.5)
					geom_leg_3.set_color(*entity.color, alpha=0.5)
					geom_leg_4.set_color(*entity.color, alpha=0.5)
				else:
					geom.set_color(*entity.color)
					geom_head.set_color(*entity.color)
				geom.add_attr(xform)
				geom_range_1.add_attr(xform_range)
				geom_range_2.add_attr(xform_range)

				geom_head.add_attr(xform_head)
				geom_leg_1.add_attr(xform_leg_1)
				geom_leg_2.add_attr(xform_leg_2)
				geom_leg_3.add_attr(xform_leg_3)
				geom_leg_4.add_attr(xform_leg_4)
				self.render_geoms.append(geom)
				self.render_geoms_xform.append(xform)
				self.render_geoms_range.append(geom_range_1)
				self.render_geoms_range.append(geom_range_2)
				self.render_geoms_xform_range.append(xform_range)
				self.render_geoms_head.append(geom_head)
				self.render_geoms_xform_head.append(xform_head)
				self.render_geoms_leg_1.append(geom_leg_1)
				self.render_geoms_xform_leg_1.append(xform_leg_1)
				self.render_geoms_leg_2.append(geom_leg_2)
				self.render_geoms_xform_leg_2.append(xform_leg_2)
				self.render_geoms_leg_3.append(geom_leg_3)
				self.render_geoms_xform_leg_3.append(xform_leg_3)
				self.render_geoms_leg_4.append(geom_leg_4)
				self.render_geoms_xform_leg_4.append(xform_leg_4)
				i += 1

			# add geoms to viewer
			for viewer in self.viewers:
				viewer.geoms = []
				for geom in self.render_geoms_range:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_head:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_leg_1:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_leg_2:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_leg_3:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_leg_4:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_hole:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_obs:
					#pdb.set_trace()
					viewer.add_geom(geom)
				for geom in self.render_geoms_snacks:
					#pdb.set_trace()
					viewer.add_geom(geom)

		results = []
		for i in range(len(self.viewers)):
			from multiagent_envs.multiagent import rendering
			# update bounds to center around agent
			cam_range = pargs.region_area + 0.2 + 1.0
			if self.shared_viewer:
				pos = np.zeros(self.world.dim_p)
			else:
				pos = self.agents[i].state.p_pos
			self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
			# update geometry positions
			ghost_num = 0
			jjj = 0
			for e, entity in enumerate(self.world.safe_holes):
				p = self.is_blocked(self.world.agents[0], self.world.obs[int(jjj/2)], jjj%2)
				self.render_geoms_hole_xform[e].set_translation(*p)
				jjj += 1
			for e, entity in enumerate(self.world.obs):
				self.render_geoms_obs_xform[e].set_translation(*entity.state.p_pos)
			for e, entity in enumerate(self.world.snacks):
				self.render_geoms_snacks_xform[e].set_translation(*entity.state.p_pos)
				if entity.is_eaten:
					self.render_geoms_snacks[e].set_color(0, 0.8, 0, alpha=0)
				else:
					self.render_geoms_snacks[e].set_color(0, 0.5, 0.5, alpha=0.5)
			for e, entity in enumerate(self.world.entities):

				if entity.state.is_ghost:
					x_loc = pargs.region_area - 0.2
					y_loc = pargs.region_area - 0.2 - 0.05 * ghost_num

					self.render_geoms_range[e * 2 + 0].set_color(0, 0.8, 0, alpha=0.5)
					self.render_geoms_range[e * 2 + 1].set_color(0, 0.8, 0, alpha=0.5)

					self.render_geoms_leg_1[e].set_color(0, 0.8, 0, alpha=0)
					self.render_geoms_leg_2[e].set_color(0, 0.8, 0, alpha=0)
					self.render_geoms_leg_4[e].set_color(0, 0.8, 0, alpha=0)
					self.render_geoms_leg_3[e].set_color(0, 0.8, 0, alpha=0)

					self.render_geoms_xform[e].set_translation(*(x_loc, y_loc))
					head_offset = np.array([math.cos(0)*entity.size, math.sin(0)*entity.size])
					self.render_geoms_xform_head[e].set_translation(*(x_loc, y_loc)+head_offset)
					ghost_num += 1


				else:
					self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
					self.render_geoms_xform_range[e].set_translation(*entity.state.p_pos)
					self.render_geoms_xform_range[e].set_rotation(entity.state.direction)
					leg_offset = np.array([math.cos(entity.state.direction + 3.14*3/4)*entity.size*.7 , math.sin(entity.state.direction + 3.14*3/4)*entity.size*.7])
					self.render_geoms_xform_leg_1[e].set_translation(*entity.state.p_pos+leg_offset)

					leg_offset = np.array([math.cos(entity.state.direction + 3.14*7/4)*entity.size*.7 , math.sin(entity.state.direction + 3.14*7/4)*entity.size*.7])
					self.render_geoms_xform_leg_2[e].set_translation(*entity.state.p_pos+leg_offset)

					leg_offset = np.array([math.cos(entity.state.direction + 3.14*1/4)*entity.size*.7 , math.sin(entity.state.direction + 3.14*1/4)*entity.size*.7])
					self.render_geoms_xform_leg_3[e].set_translation(*entity.state.p_pos+leg_offset)

					leg_offset = np.array([math.cos(entity.state.direction + 3.14*5/4)*entity.size*.7 , math.sin(entity.state.direction + 3.14*5/4)*entity.size*.7])
					self.render_geoms_xform_leg_4[e].set_translation(*entity.state.p_pos+leg_offset)

					time_angle[e] = time_angle[e] + entity.state.speed * 0.04
					if time_angle[e] > 2.3:
						time_angle[e] = time_angle[e] - 1.0

					time_angle_2 = time_angle[e] + 0.0
					if time_angle_2 > 2.3:
						time_angle_2 = time_angle_2 - 1.0

					#print(time_angle)
					self.render_geoms_xform_leg_1[e].set_rotation(entity.state.direction + time_angle[e])
					self.render_geoms_xform_leg_2[e].set_rotation(entity.state.direction - time_angle[e] +  0.7)

					self.render_geoms_xform_leg_3[e].set_rotation(entity.state.direction + time_angle_2 - 0.7)
					self.render_geoms_xform_leg_4[e].set_rotation(entity.state.direction - time_angle_2)

					if seeing[e]:
						self.render_geoms_range[e * 2 + 0].set_color(0, 0.8, 0, alpha=0.1)
						self.render_geoms_range[e * 2 + 1].set_color(0, 0.8, 0, alpha=0.1)
					else:
						self.render_geoms_range[e * 2 + 0].set_color(1, 0.8, 0, alpha=0.1)
						self.render_geoms_range[e * 2 + 1].set_color(1, 0.8, 0, alpha=0.1)

					head_offset = np.array([math.cos(entity.state.direction)*entity.size, math.sin(entity.state.direction)*entity.size])
					self.render_geoms_xform_head[e].set_translation(*entity.state.p_pos+head_offset)
			# render to display or array
			results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

		return results, time_angle

	# create receptor field locations in local coordinate frame
	def _make_receptor_locations(self, agent):
		receptor_type = 'polar'
		range_min = 0.05 * 2.0
		range_max = 1.00
		dx = []
		# circular receptive field
		if receptor_type == 'polar':
			for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
				for distance in np.linspace(range_min, range_max, 3):
					dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
			# add origin
			dx.append(np.array([0.0, 0.0]))
		# grid receptive field
		if receptor_type == 'grid':
			for x in np.linspace(-range_max, +range_max, 5):
				for y in np.linspace(-range_max, +range_max, 5):
					dx.append(np.array([x,y]))
		return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
	metadata = {
		'runtime.vectorized': True,
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, env_batch):
		self.env_batch = env_batch

	@property
	def n(self):
		return np.sum([env.n for env in self.env_batch])

	@property
	def action_space(self):
		return self.env_batch[0].action_space

	@property
	def observation_space(self):
		return self.env_batch[0].observation_space

	def step(self, action_n):
		obs_n = []
		reward_n = []
		done_n = []
		info_n = {'n': []}
		i = 0
		for env in self.env_batch:
			obs, reward, done, _ = env.step(action_n[i:(i+env.n)])
			i += env.n
			obs_n += obs
			# reward = [r / len(self.env_batch) for r in reward]
			reward_n += reward
			done_n += done
		return obs_n, reward_n, done_n, info_n

	def reset(self):
		obs_n = []
		for env in self.env_batch:
			obs_n += env.reset()
		return obs_n

	# render environment
	def render(self, mode='human', close=True):
		results_n = []
		for env in self.env_batch:
			results_n += env.render(mode, close)
		return results_n
