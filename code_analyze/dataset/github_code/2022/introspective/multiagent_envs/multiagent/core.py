import numpy as np
import math
from vars import pargs
import warnings
import pdb
warnings.filterwarnings("error")


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# physical/external base state of all entites
class EntityState(object):
	def __init__(self):
		# physical position
		self.p_pos = None
		# physical velocity
		self.p_vel = None
		self.direction = 0.0
		self.speed = 1.0
		self.is_ghost = False
		self.curr_speed = 0
		self.time_since_food = 1
		self.last_locations = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
	def __init__(self):
		super(AgentState, self).__init__()
		# communication utterance
		self.c = None

# action of the agent
class Action(object):
	def __init__(self):
		# physical action
		self.u = None
		# communication action
		self.c = None

# properties and state of physical world entity
class Entity(object):
	def __init__(self):
		# name
		self.name = ''
		# properties:
		self.size = 0.050
		# entity can move / be pushed
		self.movable = False
		# entity collides with others
		self.collide = True
		# material density (affects mass)
		self.density = 25.0
		# color
		self.color = None
		# max speed and accel
		self.max_speed = None
		self.accel = None
		# state
		self.state = EntityState()
		# mass
		self.initial_mass = 1.0

	@property
	def mass(self):
		return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
	def __init__(self):
		super(Landmark, self).__init__()

class Obstacle(Entity):
	def __init__(self):
		super(Obstacle, self).__init__()
		self.size = 0.35

class Hole(Entity):
	def __init__(self):
		super(Hole, self).__init__()
		self.size = 0.6

class Snack(Entity):
	def __init__(self):
		super(Snack, self).__init__()
		self.size = 0.05
		self.is_eaten = False

# properties of agent entities
class Agent(Entity):
	def __init__(self):
		super(Agent, self).__init__()
		# agents are movable by default
		self.movable = True
		# cannot send communication signals
		self.silent = False
		# cannot observe the world
		self.blind = False
		# physical motor noise amount
		self.u_noise = None
		# communication noise amount
		self.c_noise = None
		# control range
		self.u_range = 1.0
		# state
		self.state = AgentState()
		# action
		self.action = Action()
		# script behavior to execute
		self.action_callback = None

# multi-agent world
class World(object):
	def __init__(self):
		# list of agents and entities (can change at execution-time!)
		self.agents = []
		self.first = True
		self.landmarks = []
		self.safe_holes =[]
		self.snacks = []
		self.obs = []
		self.obs_locations = []
		# communication channel dimensionality
		self.dim_c = 0
		# position dimensionality
		self.dim_p = 2
		# color dimensionality
		self.dim_color = 3
		# simulation timestep
		self.dt = 0.05
		self.time_step = 0
		# physical damping
		self.damping = 0.25
		# contact response parameters
		self.contact_force = 1e+2
		self.contact_margin = 1e-3

	# return all entities in the world
	@property
	def entities(self):
		return self.agents + self.landmarks

	# return all agents controllable by external policies
	@property
	def policy_agents(self):
		return [agent for agent in self.agents if agent.action_callback is None]

	# return all agents controlled by world scripts
	@property
	def scripted_agents(self):
		return [agent for agent in self.agents if agent.action_callback is not None]

	# update state of the world
	def step(self):
		# set actions for scripted agents
		for agent in self.scripted_agents:
			agent.action = agent.action_callback(agent, self)
		self.integrate_state()


	# gather agent action forces
	def apply_action_force(self, p_force):
		# set applied forces
		for i,agent in enumerate(self.agents):
			if agent.movable:
				#noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
				#print(agent.action.u)
				p_force[i] = agent.action.u
		return p_force

	# gather physical forces acting on entities
	def apply_environment_force(self, p_force):
		# simple (but inefficient) collision response
		for a,entity_a in enumerate(self.entities):
			for b,entity_b in enumerate(self.entities):
				if(b <= a): continue
				[f_a, f_b] = self.get_collision_force(entity_a, entity_b)
				print(f_a)
				if(f_a is not None):
					if(p_force[a] is None): p_force[a] = 0.0
					p_force[a] = f_a + p_force[a][:2]
				if(f_b is not None):
					if(p_force[b] is None): p_force[b] = 0.0
					p_force[b] = f_b + p_force[b][:2]
		return p_force

	def angle_between_vectors(self, vector_1, vector_2):

		unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
		unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		dot_product = np.clip(dot_product, -1.0, 1.0)
		angle = np.arccos(dot_product)
		return angle

	# integrate physical state
	def integrate_state(self):
		for i,entity in enumerate(self.entities):
			if not entity.movable: continue
			entity.state.p_vel = entity.state.p_vel * (1 - self.damping)

			if entity.action.u == 0:
				entity.state.direction += entity.max_speed/4.0

			if entity.action.u == 1:
				entity.state.direction -= 0.0

			if entity.action.u == 2:
				entity.state.direction -= entity.max_speed/4.0

			if entity.action.u == 3:
				return


			entity.state.speed = entity.max_speed
			if entity.state.speed < 0:
				entity.state.speed = 0
			if entity.state.speed > entity.max_speed:
				entity.state.speed = entity.max_speed

			entity.state.p_vel = np.array([math.cos(entity.state.direction)*max(entity.state.speed, 0), math.sin(entity.state.direction)*max(entity.state.speed, 0)])

			if entity.max_speed is not None:
				speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
				if speed > entity.max_speed:
					entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
																  np.square(entity.state.p_vel[1])) * entity.max_speed

			temp = entity.state.p_pos + entity.state.p_vel * self.dt
			legal = True

			for o in self.obs:
				if np.sqrt(np.sum(np.square(temp - o.state.p_pos))) < (o.size + entity.size):
					diff = o.state.p_pos - temp
					scale = 1.0/math.sqrt(diff[0]**2 + diff[1]**2)
					diff = diff*scale*(o.size + entity.size)
					potential_spot = o.state.p_pos - diff
					if np.sqrt(np.sum(np.square(potential_spot))) < (pargs.region_area - entity.size):
						intersect = False
						for oo in self.obs:
							if np.sqrt(np.sum(np.square(potential_spot - oo.state.p_pos))) < (oo.size + entity.size):
								intersect = True
						if not intersect:
							entity.state.p_pos = o.state.p_pos - diff
					legal = False

			# if legal:
			# 	old = entity.state.p_pos
			# 	entity.state.p_pos += entity.state.p_vel * self.dt


			if pargs.walls and legal:
				dist = np.sqrt(np.sum(np.square(temp)))
				if dist > (pargs.region_area - entity.size):
					temp = temp / np.sqrt(np.square(temp[0]) + np.square(temp[1])) * (pargs.region_area - entity.size)
				intersect = False
				for oo in self.obs:
					if np.sqrt(np.sum(np.square(temp - oo.state.p_pos))) < (oo.size + entity.size):
						intersect = True
				if not intersect:
					entity.state.p_pos = temp


	def update_agent_state(self, agent):
		# set communication state (directly for now)
		if agent.silent:
			agent.state.c = np.zeros(self.dim_c)
		else:
			noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
			agent.state.c = agent.action.c + noise

	# get collision forces for any contact between two entities
	def get_collision_force(self, entity_a, entity_b):
		if (not entity_a.collide) or (not entity_b.collide):
			return [None, None] # not a collider
		if (entity_a is entity_b):
			return [None, None] # don't collide against itself
		# compute actual distance between entities
		delta_pos = entity_a.state.p_pos - entity_b.state.p_pos + .0001
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		# minimum allowable distance
		dist_min = entity_a.size + entity_b.size
		# softmax penetration
		k = self.contact_margin
		penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
		force = self.contact_force * delta_pos / dist * penetration
		force_a = +force if entity_a.movable else None
		force_b = -force if entity_b.movable else None
		return [force_a, force_b]
