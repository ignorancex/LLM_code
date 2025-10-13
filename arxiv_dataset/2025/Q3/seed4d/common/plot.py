# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import common.pose as pose


def plot_points_angles_3D(
    points: list, pitchs: list, yaws: list, folder_name, radius=1, letters=5
):
    """
    Generate a 3D plot of the given points and their corresponding pitch and yaw angles, with an optional radius and axis labels.

    Parameters:
        points (list): A list of 3D coordinates to plot.
        pitchs (list): A list of pitch angles (in radians) corresponding to each point.
        yaws (list): A list of yaw angles (in radians) corresponding to each point.
        radius (float, optional): A float representing the radius of the sphere centered at the origin. Default is 1.
        letters (int, optional): An integer representing the font size of the text labels for each point. Default is 5.

    Returns:
        None. The plot is displayed in the current matplotlib figure.
    """

    plt.rcParams["figure.figsize"] = 6, 6
    PLOT_ANGLES = True

    scaler = 0.25 * radius

    # Extract x, y, z coordinates from points
    x, y, z = map(list, zip(*points))

    # Create 3D plot
    ax = plt.figure().add_subplot(111, projection="3d")

    # Plot each point and its corresponding pitch and yaw angles
    for i in range(len(points)):
        ax.plot(
            [x[i], x[i] - radius * np.cos(pitchs[i]) * np.sin(yaws[i])],
            [y[i], y[i] - radius * np.cos(pitchs[i]) * np.cos(yaws[i])],
            [z[i], z[i] - radius * np.sin(pitchs[i])],
            "k",
        )
        if PLOT_ANGLES:
            # Plot roll angle (parallel to x-axis)
            ax.plot(
                [x[i], x[i] - scaler * np.cos(0) * np.sin(-1.57)],
                [y[i], y[i] - scaler * np.cos(0) * np.cos(-1.57)],
                [z[i], z[i] - scaler * np.sin(0)],
                "r",
            )
            # Plot yaw angle (parallel to z-axis)
            ax.plot(
                [x[i], x[i] - scaler * np.cos(1.57) * np.sin(-1.57)],
                [y[i], y[i] - scaler * np.cos(1.57) * np.cos(1.57)],
                [z[i], z[i] - scaler * np.sin(-1.57)],
                "b",
            )
            # Plot pitch angle (parallel to y-axis)
            ax.plot(
                [x[i], x[i] - scaler * np.cos(1.57) * np.sin(0)],
                [y[i], y[i] - scaler * np.cos(0) * np.cos(0)],
                [z[i], z[i] - scaler * np.sin(0)],
                "g",
            )
        ax.text(x[i], y[i], z[i], "%s" % (str(i)), size=letters, color="k")

    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=10.0, azim=230)
    plt.tight_layout()
    plt.savefig(folder_name + "/init_poses")
    # plt.show()


def gif_maker(SENSORS, TIMESTEPS, folder_name):
    for sensor_type in SENSORS:

        fig = plt.figure(figsize=(16, 13))
        columns = 3
        rows = 3
        ax = []

        if not os.path.isdir(folder_name + "/gifs/" + str(sensor_type) + "/"):
            os.makedirs(folder_name + "/gifs/" + str(sensor_type) + "/")

        for i in range(TIMESTEPS):
            for j in range(1, columns * rows + 1):
                ax.append(fig.add_subplot(rows, columns, j))
                string = (
                    folder_name
                    + "/"
                    + sensor_type
                    + "/timestep"
                    + str(i)
                    + "_cam"
                    + str(j)
                    + ".png"
                )
                img = np.asarray(Image.open(string))
                ax[-1].set_axis_off()
                fig.suptitle("CARLA", fontsize=20)
                plt.imshow(img)
            plt.subplots_adjust(
                left=0.1, bottom=0.1, right=1, top=1.2, wspace=0.01, hspace=-0.73
            )
            plt.savefig(
                folder_name + "/gifs/" + sensor_type + "/timestep" + str(i) + ".png"
            )
        plt.show()


def plot_projection_matrices(nerf_matrices, title, folder_name):

    # plot camera position and orientation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for idx, projection_matrix in enumerate(nerf_matrices):

        # decompose projection matrix into rotation, translation, and scale
        camera_matrix = np.asarray(projection_matrix)
        rotation_matrix = camera_matrix[:3, :3]
        translation_vector = camera_matrix[:3, 3]
        scale_vector = np.sqrt(np.sum(rotation_matrix**2, axis=0))

        # plot camera position as a red dot
        x, y, z = translation_vector
        ax.scatter(x, y, z, c="r", marker="o")

        # plot viewing direction as a black line
        viewing_direction = np.dot(rotation_matrix, np.array([0, 0, 1]))
        ax.quiver(x, y, z, *-viewing_direction, length=1, color="k")

        ax.text(x, y, z, "%s" % (str(idx)), size=10, color="k")

    # ax.scatter(0, 0, 0, c='b', marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(folder_name + title)

    # show plot
    # plt.show()


def plot_both_projection_matrices(
    coordinates, pitchs, yaws, origin, RADIUS, folder_name
):

    # unnormalized
    nerf_matrices_unnormalized = pose.get_OpenGL_matrices_unnormalized(
        coordinates, pitchs, yaws, origin, RADIUS
    )
    title = "/OpenGL_poses_unnormalized"
    plot_projection_matrices(nerf_matrices_unnormalized, title, folder_name)
    # normalized
    nerf_matrices_normalized = pose.get_OpenGL_matrices_normalized(
        coordinates, pitchs, yaws, origin, RADIUS
    )
    title = "/OpenGL_poses_normalized"
    plot_projection_matrices(nerf_matrices_normalized, title, folder_name)
