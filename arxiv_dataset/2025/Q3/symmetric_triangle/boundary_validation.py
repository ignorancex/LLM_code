import numpy as np
from triangle import TrilocalModel
from triangle_inequalities import (choose_W1_point, choose_W2_point, choose_W3_point,
                                   choose_W4_point, choose_W5_point, choose_GHZ_point)

c_alpha, c_beta, c_gamma = 6, 6, 6
step = 0.001
N_points = 1000
N_trials = 1000

probabilities = np.array([0.46754647, 0.08301535, 0.18701717, 0.12059518, 0.14182582])
chosen_surfaces = np.random.choice((1, 2, 3, 4, 5), size=N_points, p=probabilities)
# This code tests the W boundary. In order to test the GHZ boundary, uncomment the following line
# chosen_surfaces = np.zeros(N_points)
behaviors = np.zeros((N_points, 3, 3))
A = (1, -1)
B = (1, -1)
C = (1, -1)
A, B, C = np.meshgrid(A, B, C, indexing='ij')
# Dictionary for surface average normal vectors
normals = {0: np.array([ -0.10231534, 0.994621  , -0.01614534]),  # GHZ
           1: np.array([ 0.71834433, -0.4265963 , -0.54954256]),  # W green
           2: np.array([ 0.22677457, -0.13334915, -0.96477526]),  # W purple
           3: np.array([ 0.7565642 , -0.38813055, -0.526275  ]),  # W red
           4: np.array([ 0.7022095 , -0.05392776, -0.7099251 ]),  # W yellow
           5: np.array([ 0.40681896, -0.03642284, -0.91278243])}  # W cyan
# Dictionary for functions to generate random points on surfaces
functions = {0: choose_GHZ_point,  # GHZ
             1: choose_W1_point,   # W green
             2: choose_W2_point,   # W purple
             3: choose_W3_point,   # W red
             4: choose_W4_point,   # W yellow
             5: choose_W5_point}   # W cyan
for i, surface in enumerate(chosen_surfaces):
    # Behavior on the boundary
    behaviors[i, 1] = functions[surface]()
    E1, E2, E3 = behaviors[i, 1]
    # Calculate the maximum displacement that still yields a valid behavior
    max_step = max(abs((E1+E2-E3-1)/(np.inner((1,1,-1), normals[surface]))),
                   abs((3*E1-3*E2+E3-1)/(np.inner((3,-3,1), normals[surface]))),
                   abs((3*E1+3*E2+E3+1)/(np.inner((3,3,1), normals[surface]))))
    epsilon = min(step, max_step)
    # Behavior within the boundary
    behaviors[i, 0] = behaviors[i, 1] - epsilon*normals[surface]
    # Behavior beyond the boundary
    behaviors[i, 2] = behaviors[i, 1] + epsilon*normals[surface]

model = TrilocalModel.random(c_alpha, c_beta, c_gamma, 2, 2, 2)
rms_probability_errors = np.inf*np.ones((N_points, 3))
models = np.inf*np.ones((N_points, 3, model.degrees_of_freedom()))
for i, line in enumerate(behaviors):
    for j, point in enumerate(line):
        E1, E2, E3 = point
        p = 1/8*(1 + (A+B+C)*E1 + (A*B+B*C+A*C)*E2 + A*B*C*E3)
        model = TrilocalModel.random(c_alpha, c_beta, c_gamma, 2, 2, 2)
        model = model.optimize(p, number_of_trials=N_trials)
        rms_probability_errors[i,j] = np.sqrt(model.cost(p)/8)
        models[i,j] = model.optimizer_representation()
        
np.savez(f'W surfaces test {c_alpha}{c_beta}{c_gamma} {N_trials}.npz',
         chosen_surfaces=chosen_surfaces,
         behaviors=behaviors,
         rms_probability_errors=rms_probability_errors,
         models=models)
    