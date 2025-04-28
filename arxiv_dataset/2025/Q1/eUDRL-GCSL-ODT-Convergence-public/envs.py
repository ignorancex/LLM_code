import numpy as np

class Container :
  pass

def make_common_calc(A,S,N,G,E,goal_projection) :
  goal_projection_matrix = np.zeros([G,S])    
  for s in range(S) :
    goal_projection_matrix[goal_projection[s],s] = 1.0
  Eg = np.zeros([G,A,S])
  for s in range(S): #transition_states :
    for a in range(A) :
      Eg[:,a,s] = goal_projection_matrix.dot(E[:,a,s])

  PC_init = np.ones([S,N,G]) # initial command and state (initial eMDP) distribution
  PC_init = PC_init/PC_init.sum()
  pi_init = np.ones([A,S,N,G])/A # initial eMDP policy (we choose uniform here)


  env = Container()
  env.N = N
  env.S = S
  env.A = A
  env.E = E
  env.G = G
  env.goal_projection = goal_projection  
  env.goal_projection_matrix = goal_projection_matrix
  env.Eg = Eg
  env.PC_init_uniform = PC_init
  env.pi_init_uniform = pi_init
  env.PC_init = env.PC_init_uniform.copy()
  env.pi_init = env.pi_init_uniform.copy()

  return env




def make_env_cycle(tr_kernel_stochasticty = 0.6,N = 2,S = 3) :
  # states are organiyed in cycle
  #N = 2 # 2,4 # max horizon, variable h bellow (in the whole script) refers to (remaining horizon-1), e.g. 0 encodes horizon 1 etc.
  #S = 3 # 3,2 # num of states
  if S < 2 :
    raise Exception("This env. requires at least 2 states")
  A = 2 # num. of actions
  G = S # num of goals
  E = np.zeros([S,A,S]) # transition kernel shape (see below for actual values)
  absorbtion_states = [] # no absorptions
  transition_states = [s for s in range(S)] 
  for s in  range(S) :
    E[(s)%S,0,s] = 1-tr_kernel_stochasticty
    E[(s+1)%S,0,s] = tr_kernel_stochasticty
    E[(s)%S,1,s] = tr_kernel_stochasticty
    E[(s+1)%S,1,s] = 1-tr_kernel_stochasticty
  G = S # num of goals
  goal_space = np.arange(0,S)
  goal_projection = goal_space.astype(int)

  env = make_common_calc(A,S,N,G,E,goal_projection)
  env.tr_kernel_stochasticty = tr_kernel_stochasticty
  env.goal_space = goal_space

  return env

def make_env_bandit(tr_kernel_stochasticty = 0.6) :
  """
  A bandit, Just one initial state and horizon 1, two actions (arms), two goals
  """
  env = make_env_cycle(tr_kernel_stochasticty = tr_kernel_stochasticty ,N = 1,S = 2)
  PC_init = env.PC_init.copy()
  PC_init[1,0,:] = 0 # putting all the mass of initial extended state distribution just in state 0
  PC_init = PC_init/PC_init.sum() # renormalizing
  env.PC_init = PC_init
  return env


def make_env_bandit_matrix(trmatrix, goal_init_dist) :
  """
  assumed: S=3, G=S,A=3,N=1 the state 0 is the only initial state of original MDP, so it suffices to give transitions just from this state
  trmatrix - 3x3 (AxG) matrix, a transition kernel from state 0 given an action to a new state (goal)
  """
  if len(trmatrix.shape) != 2 or  trmatrix.shape[0] != 3 or trmatrix.shape[1] != 3 :
    raise Exception("This env. requires 3x3 matrix argument")
  N = 1 # max. horizon
  S = 3 # num. of states
  A = 3 # num. of actions
  E = np.zeros([S,A,S]) # transition kernel shape (see below for actual values)
  absorbtion_states = [] # no absorptions
  transition_states = [s for s in range(S)] 
  E[:,:,0] = trmatrix.T
  E[:,:,1] = 1/S
  E[:,:,2] = 1/S
  G = S # num of goals
  goal_space = np.arange(0,S)
  goal_projection = goal_space.astype(int)

  env = make_common_calc(A,S,N,G,E,goal_projection)
  env.tr_kernel_stochasticty = np.nan
  env.goal_space = goal_space
  # env.PC_init[0,0,1] = 0 # nonuniform in goals
  env.PC_init[0,0,:] = goal_init_dist
  env.PC_init[1] = 0
  env.PC_init[2] = 0
  env.PC_init = env.PC_init/np.sum(env.PC_init)

  return env



def make_env_line(tr_kernel_stochasticty = 0.6,S = 3, N = None) :
  # states are organiyed in cycle
  #N = 2 # 2,4 # max horizon, variable h bellow (in the whole script) refers to (remaining horizon-1), e.g. 0 encodes horizon 1 etc.
  #S = 3 # 3,2 # num of states
  N = S if N is None else N
  if S < 2 :
    raise Exception("This env. requires at least 2 states")
  A = 2 # num. of actions
  G = S # num of goals
  E = np.zeros([S,A,S]) # transition kernel shape (see below for actual values)
  absorbtion_states = [] # no absorptions
  transition_states = [s for s in range(S)] 
  for s in  range(S) :
    E[(s-1)%S,0,s] = 1-tr_kernel_stochasticty
    E[(s+1)%S,0,s] = tr_kernel_stochasticty
    E[(s-1)%S,1,s] = tr_kernel_stochasticty
    E[(s+1)%S,1,s] = 1-tr_kernel_stochasticty
  
  # disconnecting the cycle
  E[0  ,0,0  ] = 1-tr_kernel_stochasticty
  E[0  ,1,0  ] = tr_kernel_stochasticty
  E[S-1,0,0  ] = 0
  E[S-1,1,0  ] = 0
  E[0  ,1,S-1] = 0
  E[0  ,0,S-1] = 0
  E[S-1,1,S-1] = 1-tr_kernel_stochasticty
  E[S-1,0,S-1] = tr_kernel_stochasticty

  G = S # num of goals
  goal_space = np.arange(0,S)
  goal_projection = goal_space.astype(int)

  env = make_common_calc(A,S,N,G,E,goal_projection)
  env.tr_kernel_stochasticty = tr_kernel_stochasticty
  env.goal_space = goal_space

  return env


def make_env_line_start_in0 (tr_kernel_stochasticty = 0.6,S = 3,N = None) :
  """
  initial distribution resticted just to state 0
  """
  env = make_env_line(tr_kernel_stochasticty = tr_kernel_stochasticty,S = S, N = N)
  PC_init = env.PC_init_uniform.copy()
  PC_init[1:,:,:] = 0
  PC_init = PC_init/np.sum(PC_init)
  env.PC_init = PC_init
  return env



def transform_to_ODT(env,K) :
  # ODT transform
  under = env  # underlying MDP
  A = under.A
  N = under.N
  G = under.G
  K = 3 # ODT tuple len
  # 1-step neighbours from each state
  neighbours1 = np.max((under.E.sum(axis = 1) > 0).sum(axis=0))
  # K-step neighbours from each state
  neighboursK = 0
  for k in range(K) :
    neighboursK += neighbours1**k # allways maximum 2 neighbours for each state

  # enumerating all ODT states
  prefix = [-1,-1,-1]

  class Buffer :
    def __init__(self,l,w) :
      self.data = np.zeros([l,w],dtype = int)
      self.ff = 0

    def append(self,item) :
      self.data[self.ff] = item
      self.ff += 1
    
    def get(self) :
      return self.data[0:self.ff]

  ODTstates = Buffer(l = under.S*neighboursK,w = K)



  def traverse(suff_len, prefix, s ) :
    # add to ODT states
    prefix.append(s)
    ODTstates.append(np.array(prefix[-K:]))
    mask = under.E[:,:,s].sum(axis = 1)
    if suff_len-1 > 0 :
      for s_ in range(under.S) :
        if mask[s_] > 0 :
          traverse(suff_len-1, prefix, s_ )
    prefix.pop(-1)
    return

  for s in range(under.S) :
    traverse(K, prefix, s)


  ODTstates = ODTstates.get() 

  S = len(ODTstates)
  goal_projection = under.goal_projection[ODTstates[:,K-1]]
  

  
  E = np.zeros([S,A,S])
  for s in range(S) :
    us = ODTstates[s,K-1]
    for a in range(A) :
      unp = np.zeros([K],dtype = int )
      unp[0:K-1] = ODTstates[s,1:K]
      for ns in range(under.S) :
        unp[K-1] = ns        
        mask = (np.abs(ODTstates[:,0:K] - unp)).sum(axis = 1) == 0
        E[mask,a,s] = under.E[ns,a,us]



  env = make_common_calc(A,S,N,G,E,goal_projection)
  # initial distribution  
  PC_init = np.zeros([S,N,G])
  for us in range(under.S) :
      p = np.ones([K],dtype = int )*(-1)      
      p[K-1] = us
      mask = (np.abs(ODTstates[:,:] - p)).sum(axis = 1) == 0 # exactly one True index
      PC_init[mask] = under.PC_init[us]
      #print(f"p={p}, mask={mask}, under.PC_init[us] = {under.PC_init[us]}")

  # initial policy
  pi_init = np.zeros([A,S,N,G])
  for us in range(under.S) :
      mask = ODTstates[:,K-1] == us
      pi_init[:,mask,:,:] = under.pi_init[:,[us]*np.sum(mask),:,:]

  env.under = under
  env.tr_kernel_stochasticty = under.tr_kernel_stochasticty
  env.goal_space = under.goal_space
  env.K = K
  env.neighbours1 = neighbours1
  env.neighboursK = neighboursK
  env.ODTstates = ODTstates  
  env.PC_init = PC_init
  env.pi_init = pi_init
  
  return env




def make_env_gridworld(map, N, tr_kernel_stochasticty = 0.6, initial_state= None, goal_state=None) :
  """
  gridworld
  map - 2d map - a matrix like coordinate system statring from 0, 1 wall, 0 - a state
  initial_state = None - means uniform, initial_state= (i,j,h,g) tuple means koncentrate initial dist mass just in this one state
  goal_state = None - goal projection is the identity map, goal_state = (i,j) means rho(s(i,j)) = 1, rho(s') = 0 for s' \\neq s(i,j)
  """
  A = 4 # num. of actions
  action_by_strings = {"right":0,"left":1,"up":2,"down":3}
  # enumerate states 
  H,W = map.shape
  map_states = np.zeros([H,W],dtype = int)
  state_pos = np.zeros([H*W,2],dtype = int)
  s = 0
  for i in range(H) :
    for j in range(W) :
      if map[i,j] == 0 :
        map_states[i,j] = s
        state_pos[s] = np.array([i,j],dtype = int)
        s += 1
      else :
        map_states[i,j] = -1  

  state_pos = state_pos[0:s]
  S = len(state_pos)

  def is_admisible(i,j) :
    return ( 0 <= i and i < H and 0 <= j and j < W and map[i,j] == 0 )

  E = np.zeros([S,A,S]) # transition kernel shape (see below for actual values)
  for s in range(S) :
    i,j = state_pos[s]
    nsa = np.ones([A], dtype = int)*s
    if is_admisible(i,j+1) :
      nsa[0] = map_states[i,j+1]
    if is_admisible(i,j-1) :
      nsa[1] = map_states[i,j-1]
    if is_admisible(i-1,j) :
      nsa[2] = map_states[i-1,j]
    if is_admisible(i+1,j) :
      nsa[3] = map_states[i+1,j]
    for a in range(A) :
      E[nsa[a],a,s] = 1-tr_kernel_stochasticty
      other_states = np.unique(nsa[nsa != nsa[a]])
      if len(other_states) > 0 :
        E[other_states,a,s] = 1/len(other_states)*tr_kernel_stochasticty
      else : # there are no other states we have to be deterministic
        E[nsa[a],a,s] = 1
    
  if goal_state is None :
    G = S # num of goals
    goal_space = np.arange(0,S,dtype = int)
    goal_projection = goal_space.astype(int)
  else :
    G = 2
    goal_space = np.arange(0,G,dtype = int)
    goal_projection = np.zeros([S],dtype = int)
    i,j = goal_state    
    goal_projection[map_states[i,j]] = 1
  
  env = make_common_calc(A,S,N,G,E,goal_projection)
  env.tr_kernel_stochasticty = tr_kernel_stochasticty
  env.goal_space = goal_space
  env.action_by_strings = action_by_strings
  env.map_states = map_states
  env.state_pos = state_pos
  env.map = map
  env.goal_state = goal_state
  env.initial_state = initial_state
  if initial_state is not None :
    env.PC_init = np.zeros([S,N,G])
    i,j,h,g = initial_state
    env.PC_init[map_states[i,j],h,g] = 1.0

  return env


def make_env_gridworld_tiny_det(tr_kernel_stochasticty = 0.6) :
  map = np.zeros([3,3],dtype = int)
  map[1,2] = 1 # place the wall
  N = 4
  goal_state = (0,2)
  initial_state = (2,2,3,1)  
  env = make_env_gridworld(map=map, N = N,tr_kernel_stochasticty = tr_kernel_stochasticty, initial_state= initial_state, goal_state=goal_state) 
  return env

def make_env_gridworld_tiny_nondet(tr_kernel_stochasticty = 0.6) :
  map = np.zeros([3,3],dtype = int)
  map[1,2] = 1 # place the wall
  N = 4
  goal_state = (0,2)
  initial_state = (2,0,3,1)  
  env = make_env_gridworld(map=map, N = N,tr_kernel_stochasticty = tr_kernel_stochasticty, initial_state= initial_state, goal_state=goal_state) 
  return env

def make_env_gridworld_tiny(tr_kernel_stochasticty = 0.6) :
  map = np.zeros([3,3],dtype = int)
  map[1,2] = 1 # place the wall
  N = 4
  goal_state = None #(0,2)
  initial_state = None #(2,2,3,1)  
  env = make_env_gridworld(map=map, N = N,tr_kernel_stochasticty = tr_kernel_stochasticty, initial_state= initial_state, goal_state=goal_state) 
  return env


