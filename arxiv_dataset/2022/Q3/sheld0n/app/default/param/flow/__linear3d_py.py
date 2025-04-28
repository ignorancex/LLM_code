import numpy as np

u_0 = np.array([
    1.0,
    0.0,
    0.0
])
grad = np.array([
    [ 0.0,  1.0, -1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
])
a = np.array([
    0.0,
    1.0,
    1.0
])

# input:
#   x: position
#   t: time
# output:
#   u: flow_velocity
def get_velocity(x, t):
    return u_0 + grad * x + a * t;

# input:
#   x: position
#   t: time
# output:
#   j: flow_velocity_gradients
def get_velocity_gradients(x, t):
    return grad

# input:
#   x: position
#   t: time
# output:
#   a: flow_acceleration
def get_acceleration(x, t):
    return a
