import numpy as np

# input:
#   x: position
#   t: time
# output:
#   u: flow_velocity
def get_velocity(x, t):
    return np.array([
        np.cos(x[0]) * np.sin(x[1]),
        -np.sin(x[0]) * np.cos(x[1])
    ])

# input:
#   x: position
#   t: time
# output:
#   j: flow_velocity_gradients
def get_velocity_gradients(x, t):
    return np.array([
        [-np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.cos(x[1])],
        [-np.cos(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.sin(x[1])]
    ])

# input:
#   x: position
#   t: time
# output:
#   a: flow_acceleration
def get_acceleration(x, t):
    return np.array([
        0.0,
        0.0
    ])
