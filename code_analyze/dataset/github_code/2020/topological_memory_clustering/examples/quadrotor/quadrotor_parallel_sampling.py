from __future__ import print_function, division

import multiprocessing as mp
import numpy as np
import pyexotica as exo
from time import time

# Cross
OPTIMISATION_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_maze.xml' # Cross
RRT_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_rrt.xml' # Cross

# Sphere
# OPTIMISATION_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_single_sphere.xml' # Sphere
# RRT_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_rrt_single_sphere.xml' # Sphere

def run_optimization(i):
    # exo.Setup.init_ros('worker_' + str(i))
    solver = exo.Setup.load_solver(OPTIMISATION_CONFIG)
    problem = solver.get_problem()
    scene = problem.get_scene()
    mass = 0.5
    g = 9.81
    gravity_compensation_per_rotor = mass * g / 4.

    ompl = exo.Setup.load_solver(RRT_CONFIG)
    
    np.random.seed(i)
    start_state_valid = False
    start_state_original = problem.start_state.copy()
    while not start_state_valid:
        start_state = start_state_original.copy()
        start_state[0:3] += np.random.uniform(low=-1.5, high=1.5, size=(3,))
        ompl.get_problem().start_state = start_state[:6]

        # check initial state
        ompl.get_problem().get_scene().set_model_state(ompl.get_problem().start_state[:6])
        start_state_valid = ompl.get_problem().get_scene().is_state_valid()
        if not start_state_valid:
            print(i, "start state in collision, resampling")
    
    rrt_traj = ompl.solve()
    problem.start_state = start_state
    X_warmstart = problem.X.copy()
    X_warmstart[:6,:] = rrt_traj.T
    problem.X = X_warmstart

    x_start = problem.X.copy()
    x_start[:,0] = start_state
    problem.X = x_start
    problem.U = np.ones_like(problem.U) * gravity_compensation_per_rotor + 1e-2 * np.random.randn(problem.U.shape[0], problem.U.shape[1])
    # for t in range(problem.T - 1):
    #     problem.update(problem.U[:,t],t)
    
    traj = solver.solve()

    if np.any(traj > 5.1):
        print(i, "excessively large controls.")
        return None
    
    # Check distance
    tolerance = 0.02
    distance_to_target = np.linalg.norm(problem.X[:3,-1] - problem.X_star[:3,-1])
    if distance_to_target > tolerance:
        print(i, "Far away,", distance_to_target)
        return None

    # Check collision
    good_sample = True
    for t in range(problem.T - 1):
        ompl.get_problem().get_scene().update(problem.X[:6,t])
        if not ompl.get_problem().get_scene().is_state_valid():
            print(i, "timestep {0} is in collision".format(t))
            good_sample = False
            break

    if good_sample:
        print(i, "Adding sample with cost", problem.get_cost_evolution()[1][-1], problem.X[:6,-1], solver.get_planning_time(), "start_state=", start_state[:3])
        X = problem.X.copy()
        final_cost = problem.get_cost_evolution()[1][-1]
        del solver
        del scene
        del problem
        del ompl
        return (X, traj, final_cost)
    else:
        print(i, ":'(", solver.get_planning_time(), "distance=", distance_to_target)
        del solver
        del scene
        del problem
        del ompl
        return None

    # control_cost_over_time = np.zeros((problem.T,))
    # state_cost_over_time = np.zeros((problem.T,))
    # for t in range(problem.T - 1):
    #     problem.update(traj[t,:],t)
    #     state_cost_over_time[t] = problem.tau * problem.get_state_cost(t)
    #     control_cost_over_time[t] = problem.tau * problem.get_control_cost(t)
    # state_cost_over_time[-1] = problem.get_state_cost(-1)
    
    # if state_cost_over_time.sum() < 50.:
    #     print("Adding sample with cost", state_cost_over_time.sum(), problem.X[:6,-1])
    #     return (problem.X, traj, state_cost_over_time.sum())
    # else:
    #     print("Cost too high :'(", state_cost_over_time.sum(), problem.X[:6,-1])
    #     return None

if __name__ == "__main__":
    start_time = time()
    print("Starting worker pool...")
    p = mp.Pool(12)
    result = p.map(run_optimization, range(10000))
    result = np.asarray(result, dtype=object)
    print(result.shape)

    samples_X = []
    samples_U = []
    samples_final_cost = []
    for i in range(result.shape[0]):
        if result[i] is not None:
            samples_X.append(result[i][0])
            samples_U.append(result[i][1])
            samples_final_cost.append(result[i][2])

    print(np.asarray(samples_X).shape)
    np.savez('parallel_samples', samples_X=samples_X, samples_U=samples_U, samples_final_cost=samples_final_cost)
    end_time = time()
    print("Sampling complete in {0:.2f}s".format(end_time - start_time))
    p.close()
    p.join()
