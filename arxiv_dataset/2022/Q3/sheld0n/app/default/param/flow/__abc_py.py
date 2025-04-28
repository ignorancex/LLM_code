import numpy as np

# parameters
A = 1.0
B = 1.0
C = 1.0

# input:
#   x: position
#   t: time
# output:
#   u: flow_velocity
def get_velocity(x, t):
    return np.array([
        A * np.sin(x[2]) + C * np.cos(x[1]),
        B * np.sin(x[0]) + A * np.cos(x[2]),
        C * np.sin(x[1]) + B * np.cos(x[0])
    ])

# input:
#   x: position
#   t: time
# output:
#   j: flow_velocity_gradients
def get_velocity_gradients(x, t):
    return np.array([
        [0.0              , C * -np.sin(x[1]), A * np.cos(x[2]) ],
        [B * np.cos(x[0]) , 0.0              , A * -np.sin(x[2])],
        [B * -np.sin(x[0]), C * np.cos(x[1]) , 0.0              ]
    ])

# input:
#   x: position
#   t: time
# output:
#   a: flow_acceleration
def get_acceleration(x, t):
    return np.array([
        0.0,
        0.0,
        0.0
    ])
