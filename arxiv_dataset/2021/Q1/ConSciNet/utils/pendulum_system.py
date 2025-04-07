import autograd
import autograd.numpy as np
from scipy.integrate import solve_ivp

# default pen parameters same as the constant parameters used in HNNs: https://arxiv.org/abs/1906.01563
pen_params = {'g':3, # gravity
              'm':1, # mass
              'l':1} # length

def hamiltonian_fn(coords,pen_params):
    """
    Hamiltonian function
    Parameters:
    coords: (q,p)
    pen_params: dict containing the system parameters
    """
    q, p = np.split(coords,2)

    g = pen_params['g']
    m = pen_params['m']
    l = pen_params['l']

    H = m*g*l*(1-np.cos(q)) + np.square(p)/(2*m*np.square(l)) # pendulum hamiltonian

    return H

def dynamics_fn(t, coords):
    """
    Returns the time derivatives
    """
    dcoords = autograd.grad(hamiltonian_fn)(coords,pen_params)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def integrate_model(model, t_span, y0, lv,**kwargs):
    """
    Integrates model
    Parameters:
    model : a trained NN
    t_span: time interval given as a list [start_t,end_t]
    y0: y0
    lv: latent variables
    """
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32) #.view(1,2)
        hnn_in = torch.cat([x.squeeze(),lv[0]])
        
        #[(latent,q,p), (q,p)]
        dx = model.nafunc.time_derivative(hnn_in,x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
  
def get_one_trajectory(t_span = [0,10], y0 = np.array([1,0]), n_points = 50, **kwargs):
    """
    Evaluate one GT trajectory
    """
    
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    # ODE solver
    pen_sol = solve_ivp(fun=dynamics_fn, 
                           t_span=t_span, 
                           y0=y0, 
                           t_eval=t_eval, 
                           rtol=1e-10, 
                           **kwargs)

    q, p = pen_sol['y'][0], pen_sol['y'][1]

    dydt = [dynamics_fn(None, y) for y in pen_sol['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)

    return q, p, dqdt,dpdt,t_eval


def gen_data(n_to_gen = 100, len_interval = [0.3, 0.8],y0= np.array([1,0])):
    """
    Generate pendulum data
    Parameters:
    n_to_gen: number of trajectories to generate
    len_interval: length interval to sample length parameter over [len_min,len_max]
    y0: y0
    """

    x = np.zeros([n_to_gen,50])
    aux_vars = np.zeros([n_to_gen,2])
    qp_dot = np.zeros([n_to_gen,2])
    q_l = []
    p_l = []
    qdot_l = []
    pdot_l = []
    length = []
    t_eval_l = []

    for i in range(n_to_gen):
      #print(i)

      pen_params['l'] = ((len_interval[1] - len_interval[0]) * np.random.rand(1) + len_interval[0])[0]
      length.append(pen_params['l'])
      q, p, qdot,pdot, t_e = get_one_trajectory(t_span=[0,10],n_points=100, y0 = y0)
      t_to_eval = np.random.randint(0,49)


      # Encoder input
      x[i] = q[:50] # trj length 50

      # Aux latent vars
      #aux_vars[i][:,0]
      aux_vars[i] = q[50:][t_to_eval], p[50:][t_to_eval]  # q,p length 2

      # GT output
      qp_dot[i]   = qdot[0][50:][t_to_eval], pdot[0][50:][t_to_eval] # q_dot, p_dot length 2

      # GT
      q_l.append(q)
      p_l.append(p)
      qdot_l.append(qdot[0])
      pdot_l.append(pdot[0])
      t_eval_l.append(t_to_eval)

    data = {'x': x,
          'aux_vars': aux_vars,
          'qp_dot': qp_dot,
          'q': q_l,
          'p': p_l,
          'qdot': qdot_l,
          'pdot': pdot_l,
          't_eval': t_eval_l,
          'length': length}

    return data


def add_noies(trj,sigma = 0.03):
    noies = np.random.normal(0,sigma, len(trj))
    return trj + noies

def noies_data(data,sigma = 0.03):
    q_with_noies = [add_noies(q, sigma) for q in data['q']]
    p_with_noies = [add_noies(p, sigma) for p in data['p']]
    data['q_with_noies'] = q_with_noies
    data['p_with_noies'] = p_with_noies
    return data

def compute_derivs(data):
    qdot_noies = [np.gradient(q,0.1) for q in data['q_with_noies']]
    pdot_noies = [np.gradient(p,0.1) for p in data['p_with_noies']]
    data['qdot_with_noies'] = qdot_noies
    data['pdot_with_noies'] = pdot_noies
    return data

def get_aux_vars(data):
    """
    sample auxillary latent variables from data
    """
    n_to_gen = data['x'].shape[0]
    aux_vars = np.zeros([n_to_gen,2])
    qp_dot1 = np.zeros([n_to_gen,2])

    for i in range(n_to_gen):
      t_to_eval = np.random.randint(0,49)

      # Aux latent vars
      aux_vars[i] = data['q_with_noies'][i][50:][t_to_eval], data['p_with_noies'][i][50:][t_to_eval]  # q,p length 2

      # GT output
      qp_dot1[i,0]   = data['qdot_with_noies'][i][50:][t_to_eval] #, data['pdot_with_noies'][50:][t_to_eval]) # q_dot, p_dot length 2
      qp_dot1[i,1]   = data['pdot_with_noies'][i][50:][t_to_eval]

    data['qp_dot_with_noies'] = qp_dot1
    data['aux_vars_with_noies'] = aux_vars
    return data


def setup_trial(data,sigma):
    """
    Setup a trial for a given noise level
    Parameters:
    data: data dict
    sigma: noise level
    """
    trial_data = noies_data(data, sigma) 
    trial_data = compute_derivs(trial_data)
    trial_data = get_aux_vars(trial_data)

    # config q as input trajectory
    x_q = np.zeros([50000,100])
    idx = 0
    for q in trial_data['q_with_noies']:
      x_q[idx] = q 
      idx += 1

    #trial_data['x'] = q_p
    trial_data['data_in'] = np.hstack([x_q,trial_data['aux_vars_with_noies']])
    trial_data['n_level'] = sigma

    return trial_data
