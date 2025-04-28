# Classical and intention avare UDRL algorithms.
# Theoretical versions (providing theoretical distributions) as well as implementation
# versions (providind sample approximations of theoretical distributions) of algorithms are provided.
# Implementation versions are intended mainly for verification purposes.
# Various helper functions are also provided for calculating:
# (cross) values given eMDP policy,
# state visitation distribution given eMDP policy, and initial state-command joint
# optimal eMDP policy
# etc.

import numpy as np

from envs import Container


def pi_values(env,pi) :
  """
  Computation of values Q,V for given policy pi computation
  """
  A,S,N,G = env.A, env.S, env.N, env.G
  E,Eg = env.E,env.Eg  

  Q = np.zeros([A,S,N,G])
  V = np.zeros([S,N,G]) # now all the absorbtions are initialized to 0 (correct value)
  h = 0
  for s in range(S) :
    for g in range(G) :
      Q[:,s,h,g] = Eg[g,:,s]
      V[s,h,g] = np.inner( Q[:,s,h,g], pi[:,s,h,g] )
  for h in range(1,N) :
    for s in range(S) :      
      for g in range(G):
        Q[:,s,h,g] = V[:,h-1,g].dot(E[:,:,s])
        V[s,h,g] = np.inner( Q[:,s,h,g], pi[:,s,h,g] )
  return (Q,V)




def optimum(env) :
  """
  Optimum Q,V,pi computation
  """
  A,S,N,G = env.A, env.S, env.N, env.G
  E,Eg = env.E,env.Eg  

  Q = np.zeros([A,S,N,G])
  V = np.zeros([S,N,G])
  pi = np.ones([A,S,N,G])/A # redefined by computation
  h = 0
  for s in range(S) :
    for g in range(G) :
      Q[:,s,h,g] = Eg[g,:,s]
      if np.sum(Q[:,s,h,g]) > 0 :        
        pi[:,s,h,g] = (Q[:,s,h,g] == max(Q[:,s,h,g])).astype(float)
        pi[:,s,h,g] = pi[:,s,h,g]/pi[:,s,h,g].sum()
        V[s,h,g] = np.inner( Q[:,s,h,g], pi[:,s,h,g] )
      else:
        pi[:,s,h,g] = 1/A # when there is no value the optimum support should be whole A, so pi(M(\bar{s})|\bar{s} = 1)
        V[s,h,g] = 0  
  for h in range(1,N) :
    for s in range(S) :        
      for g in range(G):
        Q[:,s,h,g] = V[:,h-1,g].dot(E[:,:,s])
        if np.sum(Q[:,s,h,g]) > 0 :
          pi[:,s,h,g] = (Q[:,s,h,g] == max(Q[:,s,h,g]) ).astype(float)
          pi[:,s,h,g] = pi[:,s,h,g]/pi[:,s,h,g].sum()
          V[s,h,g] = np.inner( Q[:,s,h,g], pi[:,s,h,g] )
        else:          
          pi[:,s,h,g] = 1/A
          V[s,h,g] = 0        
  Opt = Container()
  Opt.pi = pi
  Opt.pi_supp = pi > 0 # pi is nonzero on every optimal action
  Opt.V = V
  Opt.Q = Q
  return Opt


  





def pi_vist(env,pi,PC_init) :  
  """
  state visitation distribution
  """
  A,S,N,G = env.A, env.S, env.N, env.G
  E = env.E

  # eMDP state distribution after ss steps [ss,S,N,G]  
  Ps = np.zeros([N,S,N,G])
  st = 0
  Ps[st,:,:,:] = PC_init
  for st in range(1,N) :
    for s in range(S) :
      for h in range(0,N-1) :
        Ps[st,s,h,:] = np.zeros([G])
        # for s_ in range(S): 
        #   for a in range(A):
        #     Ps[st,s,h,:] += Ps[st-1,s_,h+1,:]*pi[a,s_,h+1,:]*E[s,a,s_]
        # The commented code above was replaced by faster version below during profiling. However, we left the commented version for better readability
        for a in range(A):
          Ps[st,s,h,:] += (Ps[st-1,:,h+1,:]*pi[a,:,h+1,:]*(E[s,a,:][:,None])).sum(axis=0)
      Ps[st,s,N-1,:] = np.zeros([G])


  rho = np.zeros([S,N,G])
  for s in range(S) :
    for h in range(N) :
      for st in range(N-h) :
        rho[s,h,:] += Ps[st,s,h,:]
  rho = rho/rho.sum()
  return rho




def cross_pi_values(env,pi) :
  """
  These 'cross' Values appear naturaly when one tries to write down classical UDRL recursion.
  Simply the command we are interested to achieve (to observe - we compute probability for) is different from the intended one.
  CQ - cross Q-value
  CV - cross V-value
  """
  A,S,N,G = env.A, env.S, env.N, env.G
  E,Eg = env.E,env.Eg

  CQ = np.zeros([A,N,G,S,N,G]) # (a,h,g,s,h',g')  h,g - observed, h',g' - intended
  CV = np.zeros([N,G,S,N,G])
  h = 0
  for s in range(S) :
    for g in range(G) :
      for h_ in range(N) :
        for g_ in range(G) :
          CQ[:,h,g,s,h_,g_] = Eg[g,:,s]
          CV[h,g,s,h_,g_] = np.inner( CQ[:,h,g,s,h_,g_], pi[:,s,h_,g_] )
  
  for h in range(1,N) :
    for s in range(S) :
      for g in range(G) :
        for h_ in range(h,N) :
          for g_ in range(G) :
            CQ[:,h,g,s,h_,g_] = CV[h-1,g,:,h_-1,g_].dot(E[:,:,s])
            CV[h,g,s,h_,g_] = np.inner( CQ[:,h,g,s,h_,g_], pi[:,s,h_,g_] )
  return (CQ,CV)






def classical_udrl_theory2(env,NconvT,pi_init=None,PC_init=None,SegSpace=None,epsilon=0.0) :
  """
  classical UDRL version, computation of theoretical distributions - simplified version :
  CQ - cross Q-value
  CV - cross V-value
  pi - final eMDP policy
  rho - state visitation dist
  This 'classical' version of UDRL ignores intended command in behaviour comupation in opposite to
  intended command avare version
  """
  A,S,N,G = env.A, env.S, env.N, env.G
  pi_init = env.pi_init if pi_init is None else pi_init
  PC_init = env.PC_init if PC_init is None else PC_init
  #print(f"pi_init={pi_init}")
  #print(f"PC_init={PC_init}")
  SegSpace = "all" if SegSpace is None else SegSpace
  SegSpaces = ["all","trailing","diag"]
  if SegSpace not in SegSpaces :
    raise exception("SegSpace should be one of {SegSpaces} and it is {SegSpace}")
  
  #clasic UDRL
  pi_saved = np.zeros([NconvT,A,S,N,G])
  rho_saved = np.zeros([NconvT,S,N,G])
  J_saved = np.zeros([NconvT])
  pi = pi_init.copy()
  for n in range(NconvT) :
    rho = pi_vist(env,pi,PC_init)
    (CQ,CV) = cross_pi_values(env,pi)
    Q,V = pi_values(env,pi) # just for J computation
    J = np.sum(env.PC_init*V) # just for J computation
    pi_new = np.zeros(pi.shape)
    for s in range(S) :
      for h in range(N) :
        for g in range(G) :
          h_range = range(h,N)
          g_range = range(G)
          if SegSpace == "trailing" :
            h_range = range(h,h)
          if SegSpace == "diag" :
            h_range = range(h,h)
            g_range = range(g,g)
          for h_ in h_range :
            for g_ in g_range :
              pi_new[:,s,h,g] += CQ[:,h,g,s,h_,g_]*pi[:,s,h_,g_]*rho[s,h_,g_]
          if pi_new[:,s,h,g].sum() > 0 :
            pi_new[:,s,h,g] = pi_new[:,s,h,g]/(pi_new[:,s,h,g].sum())
            # regularization (turned off for epsilon == 0)
            pi_new[:,s,h,g] = (1-epsilon)*pi_new[:,s,h,g] + epsilon/A
          else : #use uniform
            pi_new[:,s,h,g] = 1.0/A
    rerr = np.max(np.abs(pi_new-pi))/np.max(pi)
    pi = pi_new
    #if env.do_sym_checks :
    #  sym_check(env,CQ,rho,pi)
    pi_saved[n] = pi
    rho_saved[n] = rho
    J_saved[n] = J

  CUDRLT2 = Container()
  CUDRLT2.rho = rho
  CUDRLT2.pi = pi
  CUDRLT2.CQ = CQ
  CUDRLT2.CV = CV
  CUDRLT2.pi_saved = pi_saved
  CUDRLT2.rho_saved = rho_saved
  CUDRLT2.J_saved = J_saved
  CUDRLT2.pi_init = pi_init
  return CUDRLT2



# utility function regarding bounds computation


def test_mu_supp(env_det,the_interesting_states) :
  """
  testing aplicability of x_star bound
  $\the_nteresting_states \in supp \bar{mu}$
  """
  return 0 == np.sum( the_interesting_states != np.logical_and(the_interesting_states, env_det.PC_init > 0) )

def test_M1(env_det, the_interesting_states, Opt_det) :
  """
  testing aplicability of x_l x_u bound
  $(\forall \bar{s} \in \the_nteresting_states) : M(\bar(s)) = 1 $
  """
  # a policy : [A,S,N,G]
  # the_interesting_states : [S,N,G]
  supp_lens = Opt_det.pi_supp.astype(int).sum(axis=0)  
  return 0 == np.sum(supp_lens[the_interesting_states] != 1)


def min_piM_projection(piar,Opt,the_interesting_states) :
  """
  Calculating $min_{s\in \bar{S}(\lambda_0)} pi(M(\bar{s})|\bar{s})$
  """  
  M = (Opt.pi_supp).astype(float)
  action_dim = len(piar.shape)-4
  min_piM = (piar*M).sum(axis=action_dim)[...,the_interesting_states].min(axis = action_dim) # .min(axis=tuple(np.arange(action_dim,action_dim+3,dtype = int)))
  return min_piM

def norm1_dist(lambda0,lambda1) :
  """
  Norm 1 distance between transition kernels
  """
  new_state_dim = 0
  dist = np.abs((lambda0-lambda1)).sum(axis=new_state_dim).max()
  return dist

def x_star(gamma) : # (beta,alpha)
  if gamma < 1 and gamma > 0  :
    xs = 1 - gamma
  else :
    xs = np.nan
  return xs

def get_gamma(alpha,beta,betas) :
  return betas/((1-beta)*alpha)

def get_alpha_beta_nu(env,env_det) :
  N = env.N
  min_mu = np.min(env.PC_init[env.PC_init > 0])
  alpha = 2*min_mu/(N*(N+1))
  delta = norm1_dist(env.E,env_det.E)
  beta = np.zeros([N+1])
  nu = np.zeros([N+1])
  gamma = np.zeros([N+1])
  betas = N*delta/2
  for h in range(1,N+1) :
    if h == 1 :
      beta[h] = np.max([delta,betas])
    else :
      beta[h] = delta + nu[h-1] + beta[h-1]
    if beta[h] >= 1 :
      beta[h] = np.nan
    gamma[h] =  get_gamma(alpha,beta[h],betas)
    if gamma[h] >= 1 :
      gamma[h] = np.nan
    nu[h] = 2*(1-x_star(gamma[h]))
  betaN = beta[N]
  gammaN = gamma[N]
  nuN = nu[N]
  return alpha, betas, beta, nu, gamma, betaN, nuN, gammaN # alpha, beta, nu, betaN


def reg_get_gamma(alpha,beta,betas,epsilon,N) :
  return betas/(((1-epsilon)**N-beta)*alpha)

def reg_get_alpha_beta_nu(env,env_det,epsilon) :
  N = env.N
  A = env.A
  min_mu = np.min(env.PC_init[env.PC_init > 0])  
  delta = norm1_dist(env.E,env_det.E)
  alpha = 2*min_mu/(N*(N+1))*(epsilon/A)**N*(1-delta/2)**N
  beta = np.zeros([N+1])
  nu = np.zeros([N+1])
  gamma = np.zeros([N+1])
  betas = N*delta/2
  for h in range(1,N+1) :
    if h == 1 :
      beta[h] = np.max([delta,betas])
    else :
      beta[h] = delta + nu[h-1] + beta[h-1]
    if beta[h] >= 1 :
      beta[h] = np.nan
    gamma[h] =  reg_get_gamma(alpha,beta[h],betas,epsilon,N)
    if gamma[h] >= 1 :
      gamma[h] = np.nan
    nuM = np.zeros([A])
    xsM = np.zeros([A])
    for M in range(1,A+1) :
      xsM[M-1] = reg_get_xstar(gamma[h],epsilon,M,A)
      nuM[M-1] = 2*(1-epsilon*(1-M/A)-gamma[h] - xsM[M-1] )
    nu[h] = np.max(nuM)
    xsmin = min(xsM)
  betaN = beta[N]
  gammaN = gamma[N]
  nuN = nu[N]
  return alpha, betas, beta, nu, gamma, betaN, nuN, gammaN, xsmin # alpha, beta, nu, betaN



def get_contraction_fixed_point(contraction, lb, ub, eps = 1e-3, halt_time = 1000) :  
  xa = ub - eps
  xb = lb + eps
  it = 0
  while np.abs(xa-xb) > eps :
    xa = contraction(xa)
    xb = contraction(xb)
    it += 1
    if it > 1000 :
      raise Exception("Slow convergence")
  return (xa+xb)/2


def get_b0(N) :
  return ( 1/(2*N)*((2*N-1)/(2*N))**(2*N-1) )

def get_b_xl_xu_from_env(env,env_det) :  
  N = env.N
  min_mu = np.min(env.PC_init[env.PC_init > 0])  
  delta = norm1_dist(env.E,env_det.E)  
  if delta >= 2.0 :
    return np.nan, np.nan, np.nan
  b = delta*(N**2)*(N+1)/(4*((1-delta/2)**(2*N))*min_mu)
  xl, xu = get_xl_xu(N,b)
  return b, xl, xu 


def get_xl_xu(N,b) :
  b0 = get_b0(N)
  if b0 <= b :
    return np.nan, np.nan
  def g1(x) :
    return -b/(x**(2*N-1)) + 1
  xu = get_contraction_fixed_point(contraction = g1, lb = (2*N-1)/(2*N), ub = 1) 
  def g2(x) :
    return (b/(1-x))**(1/(2*N-1))
  xl = get_contraction_fixed_point(contraction = g2, lb = 0, ub = (2*N-1)/(2*N))    #lb = (2*N-1)/(2*N), ub = 1)   
  return xl, xu


def get_the_interesting_states(env_det) :
  Opt_det = optimum(env_det)
  rho_det = pi_vist(env_det,env_det.pi_init_uniform,env_det.PC_init) # S,N,G
  supp_d00 = (Opt_det.V*rho_det.sum(axis = (1,2),keepdims = True)) > 0 
  the_interesting_states = (supp_d00*rho_det) > 0
  return the_interesting_states, Opt_det, rho_det, supp_d00


def reg_get_xstar(gamma,epsilon,M,A) :  
  if (epsilon <= 0) or (gamma <= 0) :    
    return np.nan
  if 1 <= epsilon + gamma :
    return np.nan
  
  xsh = 1 - epsilon*(1-M/A)-gamma
  xs = (xsh + (xsh**2 + 4*gamma*epsilon*M/A)**0.5)/2
  return xs

