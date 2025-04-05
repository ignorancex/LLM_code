import numpy as np

def generate_forward(lambda_list,t_list,diff_const=0.5,k_spring=2):
    """Generate particle trajectories (X_t) given the control sequence (C_t)

    Args:
        lambda_list (array): the array containing the control sequence for different samples
        t_list (array): the array containg the simulation times 
        diff_const (float, optional): diffusion constant. Defaults to 0.5.
        k_spring (int, optional): spring constant. Defaults to 2.

    Returns:
        array: Trajectories for each control sequence. 
    """
    num_particles = lambda_list.shape[0]
    num_steps = lambda_list.shape[1]-1
    xs = np.zeros((num_particles,num_steps+1))
    xs[:,0] = diff_const/k_spring*np.random.randn(num_particles)  
    dt = t_list[1]
    std = np.sqrt(2*diff_const*dt)
    for i in range(num_steps):
        delta_w = std*np.random.randn(num_particles)
        xs[:,i+1] = xs[:,i]*(1-k_spring*dt)+k_spring*lambda_list[:,i+1]*dt+delta_w
    return xs
def gen_rand_lambda(t_list,num_trajs,k_gamma=1.,t_gamma = 1.):
    """Generate control sequences with switching times drawn from the gamma distribution

    Args:
        t_list (array): the array containg the simulation times .
        num_trajs (int): number of sample sequences
        k_gamma (float, optional): the shape of the Gamma distribution. Defaults to 1.
        t_gamma (float, optional): the mean switching time. Defaults to 1.

    Returns:
        _type_: _description_
    """
    tau = t_list[1]-t_list[0]
    num_t = t_list.shape[0]
    lambda_t=np.zeros((num_trajs,num_t))
    flag = np.random.choice([-1,1],num_trajs)
    lambda_t[:,0] = flag
    current_t = np.zeros(num_trajs,dtype='int')
    while np.min(current_t)!=num_t:
        ind_jump = np.min([(np.random.gamma(k_gamma,t_gamma/k_gamma,num_trajs)
                            /tau).astype('int'),(num_t-current_t)],axis=0)
        for i in range(num_trajs):
            lambda_t[i,current_t[i]:current_t[i]+ind_jump[i]] = flag[i]
        current_t += ind_jump 
        flag *= -1
    return lambda_t 