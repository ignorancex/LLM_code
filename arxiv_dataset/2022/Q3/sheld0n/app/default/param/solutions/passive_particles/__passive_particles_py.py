import numpy as np

# input:
#   state: particle state variable (position)
#   u: flow velocity at the particle position
# output:
#   dstate: temporal derivative of the particle state variable (its velocity in this case)
def state_temporal_derivative(state, u):
    return u

# input:
#   particle_state_size: dimension of the state variable of one particle
#   particle_number: number of particle
# output:
#   initial particle state
def init(particle_state_size, particle_number):
    # initialize output
    initial_particles_state = np.empty(shape=(particle_number, particle_state_size))
    # set random initial positions to all particles
    for particle_index in range(particle_number):
        initial_particles_state[particle_index, :] = np.random.default_rng().uniform(-np.pi, np.pi, size=particle_state_size)
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
    # extract format number
    format_number = particle_number//10 + 1
    # post process the position of each particle
    particles_state = state.reshape((particle_number, particle_state_size))
    for particle_index in range(particle_number):
        x = particles_state[particle_index, :]
        for i in range(particle_state_size):
            post_processing["passive_particles__index_{:0>{}d}__pos_{}".format(particle_index, format_number, i)] = x[i]
    # return output
    return post_processing
