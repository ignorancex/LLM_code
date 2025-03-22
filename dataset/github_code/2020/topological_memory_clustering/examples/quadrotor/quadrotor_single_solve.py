from __future__ import print_function, division

import numpy as np
import pyexotica as exo
import exotica_ddp_solver_py
from pyexotica.publish_trajectory import plot
from time import time, sleep
import matplotlib.pyplot as plt

OPTIMISATION_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_maze.xml' # Cross
# OPTIMISATION_CONFIG = '{topological_memory_clustering}/examples/quadrotor/resources/quadrotor_single_sphere.xml' # Sphere

if __name__ == "__main__":
    exo.Setup.init_ros()
    solver = exo.Setup.load_solver(OPTIMISATION_CONFIG)
    problem = solver.get_problem()
    scene = problem.get_scene()
    cs = scene.get_collision_scene()
    kt = scene.get_kinematic_tree()
    vis = exo.VisualizationMoveIt(scene)
    mass = 0.5
    g = 9.81
    gravity_compensation_per_rotor = mass * g / 4.

    # start_state = problem.start_state.copy()
    # start_state[0:3] += np.random.uniform(low=-2.5,high=2.5,size=(3,))
    # problem.start_state = start_state
    # x_start = problem.X.copy()
    # x_start[:,0] = start_state
    # problem.X = x_start
    # problem.U = np.ones_like(problem.U) * gravity_compensation_per_rotor + 1e-2 * np.random.randn(problem.U.shape[0], problem.U.shape[1])
    problem.U = np.ones_like(problem.U) * gravity_compensation_per_rotor + 0.01 * np.random.uniform(low=-1.,high=1.,size=(problem.U.shape[0], problem.U.shape[1]))
    # for t in range(problem.T - 1):
    #     problem.update(problem.U[:,t],t)
    
    traj = solver.solve()
    print("Solved in", solver.get_planning_time())
    
    control_cost_over_time = np.zeros((problem.T,))
    state_cost_over_time = np.zeros((problem.T,))
    for t in range(problem.T - 1):
        problem.update(traj[t,:],t)
        state_cost_over_time[t] = problem.tau * problem.get_state_cost(t)
        control_cost_over_time[t] = problem.tau * problem.get_control_cost(t)
    state_cost_over_time[-1] = problem.get_state_cost(-1)

    plt.figure(1)
    plt.plot(problem.get_cost_evolution()[1], label='Cost')
    plt.plot(solver.control_cost_evolution, label='Control Cost')
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, len(solver.control_cost_evolution)-1)

    plt.figure(2)
    plt.plot(traj)
    plt.xlim(0, problem.T - 1)
    plt.show()


    good_sample = True

    # Check it's close to the target
    tolerance = 0.01
    distance_to_target = np.linalg.norm(problem.X[:3,-1] - problem.X_star[:3,-1])
    if distance_to_target > tolerance:
        print("Far away,", distance_to_target)
        good_sample = False

    # Check that it's collision-free and that we are close-ish to the goal
    # for i in range(problem.T - 1):
    #     scene.update(problem.X[:6,i])
    #     if not scene.is_state_valid():
    #         print(i, "is no good")
    #         good_sample = False
    
    # if state_cost_over_time.sum() < 50.:
    #     print("Adding sample with cost", state_cost_over_time.sum(), problem.X[:6,-1])
    #     # return (problem.X, traj, state_cost_over_time.sum())
    # else:
    #     print("Cost too high :'(", state_cost_over_time.sum(), problem.X[:6,-1])
    #     return None
    # plot(problem.get_cost_evolution()[1])
    # plot(state_cost_over_time)

    kt.publish_frames()
    vis.display_trajectory(problem.X[:6,:].transpose())
    sleep(1)

