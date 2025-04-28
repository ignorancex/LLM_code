import numpy as np

# parameters
aspect_ratio = 1.0
factor = (aspect_ratio*aspect_ratio - 1.0) / (aspect_ratio*aspect_ratio + 1.0) # computed based on the aspect_ratio
swimming_velocity = 0.5

# input:
#   state: particle state variable (position)
#   u: flow velocity at the particle position
# output:
#   dstate: temporal derivative of the particle state variable (its velocity in this case)
def state_temporal_derivative(state, u, grad_u):
    # extract dim from statesize
    dim = state.size//2
    # extracting input
    x = state[0:dim].reshape((dim, 1))
    p = state[dim:state.size].reshape((dim, 1))
    _grad_u = grad_u.reshape((dim, dim)).transpose() # must be transposed because anotherconvention is used in C++
    sym_grad_u = 0.5 * (_grad_u + _grad_u.transpose())
    skew_grad_u = 0.5 * (_grad_u - _grad_u.transpose())
    # initializing output
    dstate = np.empty(state.shape)
    dx = dstate[0:dim]
    dp = dstate[dim:state.size]
    # equation
    dx[:] = u + swimming_velocity * (p / np.linalg.norm(p)).flatten()
    dp[:] = (skew_grad_u @ p + factor * (sym_grad_u @ p - (p.transpose() @ (sym_grad_u @ p)) * p)).flatten();
    # return state
    return dstate

# input:
#   particle_state_size: dimension of the state variable of one particle
#   particle_number: number of particle
# output:
#   initial particle state
def init(particle_state_size, particle_number):
    # initialize output
    initial_particles_state = np.empty(shape=(particle_number, particle_state_size))
    # extract dim from statesize
    dim = particle_state_size//2
    # set random initial positions to all particles
    for particle_index in range(particle_number):
        x = initial_particles_state[particle_index, 0:dim]
        p = initial_particles_state[particle_index, dim:particle_state_size]
        x[:] = np.random.default_rng().uniform(-np.pi, np.pi, size=dim)
        p[:] = np.random.default_rng().normal(size=dim)
        p[:] = p / np.linalg.norm(p)
    # return output
    return initial_particles_state.flatten()

# input:
#   state: state variable that groups the state variables of all particles
#   particle_state_size: dimension of the state variable of one particle
#   particle_number: number of particle
# output:
#   dictionary that contains the post processed data
def post(state, particle_state_size, particle_number):
    # initialize output
    post_processing = {}
    # extract dim from statesize
    dim = particle_state_size//2
    format_number = particle_number//10 + 1
    # post process the position of each particle
    particles_state = state.reshape((particle_number, particle_state_size))
    for particle_index in range(particle_number):
        x = particles_state[particle_index, 0:dim]
        for i in range(dim):
            post_processing["passive_particles__index_{:0>{}d}__pos_{}".format(particle_index, format_number, i)] = x[i]
        p = particles_state[particle_index, dim:particle_state_size]
        for i in range(dim):
            post_processing["passive_particles__index_{:0>{}d}__p_{}".format(particle_index, format_number, i)] = p[i]
    # return output
    return post_processing
