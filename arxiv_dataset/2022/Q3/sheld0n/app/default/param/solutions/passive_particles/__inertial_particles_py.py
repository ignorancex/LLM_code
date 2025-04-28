import numpy as np

# parameters
reaction_time = 1.0 

# input:
#   state: particle state variable (position)
#   u: flow velocity at the particle position
# output:
#   dstate: temporal derivative of the particle state variable (its velocity in this case)
def state_temporal_derivative(state, u):
    # extract dim from statesize
    dim = state.size//2
    # extracting input
    x = state[0:dim]
    v = state[dim:state.size]
    # initializing output
    dstate = np.empty(state.shape)
    dx = dstate[0:dim]
    dv = dstate[dim:state.size]
    # equation
    dx[:] = v
    dv[:] = (u - v) / reaction_time
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
        v = initial_particles_state[particle_index, dim:particle_state_size]
        x[:] = np.random.default_rng().uniform(-np.pi, np.pi, size=dim)
        v[:] = 0.0
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
        v = particles_state[particle_index, dim:particle_state_size]
        for i in range(dim):
            post_processing["passive_particles__index_{:0>{}d}__v_{}".format(particle_index, format_number, i)] = v[i]
    # return output
    return post_processing
