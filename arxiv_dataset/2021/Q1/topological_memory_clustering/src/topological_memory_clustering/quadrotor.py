from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_state_trajectories_3d(X_to_plot, color=None, alpha=0.5, fig=None):
    if fig is None:
        fig = plt.figure()
    fig.set_size_inches(12,12)
    ax = plt.axes(projection='3d')
    print(X_to_plot.shape)
    for i in range(X_to_plot.shape[0]):
        if color is None:
            ax.plot3D(X_to_plot[i,0,:], X_to_plot[i,1,:], X_to_plot[i,2,:], alpha=alpha)
        else:
            ax.plot3D(X_to_plot[i,0,:], X_to_plot[i,1,:], X_to_plot[i,2,:], color=color[i], alpha=alpha)

    # Bars
    tmp_values = np.arange(-5,5.25,.25)
    tmp_zeros = np.zeros_like(tmp_values)
    obstacle_color = 'black'
    ax.plot3D(tmp_zeros, tmp_values, tmp_zeros, color=obstacle_color, linewidth=5, alpha=0.5)
    ax.plot3D(tmp_values, tmp_zeros, tmp_zeros, color=obstacle_color, linewidth=5, alpha=0.5)
    ax.plot3D(tmp_zeros, tmp_zeros, tmp_values, color=obstacle_color, linewidth=5, alpha=0.5)

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    return fig

def plot_control(samples_U, control_limit_low=None, control_limit_high=None, fig=None):
    # if fig is None:
    #     fig = plt.figure()
    #fig.set_size_inches(12,12)

    sample_dim = samples_U.shape[0]
    for i in range(sample_dim):
        plt.plot(samples_U[i,:,:], label=i)

    plt.xlim(0,samples_U.shape[1]-1)
    plt.ylim(-1,6)
    plt.xlabel('Timestep')
    plt.ylabel(r'Control ($N$)')
    if control_limit_low is not None:
        plt.axhline(y=control_limit_low, color='r', linestyle='--')
    if control_limit_high is not None:
        plt.axhline(y=control_limit_high, color='r', linestyle='--')
    plt.title('Control trajectories')
    plt.tight_layout()
    # return fig

def solve_oc_problem(start_state=None, U_init=None, U_noise=1e-3, debug=False):
    import pyexotica as exo
    solver = exo.Setup.load_solver('{topological_memory_clustering}/quadrotor/resources/quadrotor_maze.xml')
    problem = solver.get_problem()
    scene = problem.get_scene()
    mass = 0.5
    g = 9.81
    gravity_compensation_per_rotor = mass * g / 4.

    if start_state is None:
        start_state_valid = False
        start_state_original = problem.start_state.copy()
        while not start_state_valid:
            start_state = start_state_original.copy()
            start_state[0:3] += np.random.uniform(low=-2.5,high=2.5,size=(3,))
            problem.start_state = start_state

            # check initial state
            scene.set_model_state(problem.start_state[:6])
            start_state_valid = scene.is_state_valid()
            if not start_state_valid:
                print(i, "start state in collision, resampling")
    else:
        # TODO: Implement
        assert False

    x_start = problem.X.copy()
    x_start[:,0] = start_state
    problem.X = x_start
    if U_init is None:
        problem.U = np.ones_like(problem.U) * gravity_compensation_per_rotor + U_noise * np.random.randn(problem.U.shape[0], problem.U.shape[1])
    else:
        problem.U = U_init
    
    traj = solver.solve()
    debug and print("OC problem solved in", solver.get_planning_time())

    if np.any(traj > 5.5):
        print("Failed - Excessively large controls.")
        return None
    
    # Check distance
    tolerance = 0.05
    distance_to_target = np.linalg.norm(problem.X[:3,-1] - problem.X_star[:3,-1])
    if distance_to_target > tolerance:
        debug and print("Failed - Far away,", distance_to_target)
        return None

    # Check collision
    good_sample = True
    for t in range(problem.T - 1):
        scene.update(problem.X[:6,t])
        if not scene.is_state_valid():
            good_sample = False

    if good_sample:
        debug and print("Converged - cost", np.min(problem.get_cost_evolution()[1]), problem.X[:6,-1])
        return (problem.X, traj, np.min(problem.get_cost_evolution()[1]))
    else:
        debug and print("Failed - in collision")
        return None
