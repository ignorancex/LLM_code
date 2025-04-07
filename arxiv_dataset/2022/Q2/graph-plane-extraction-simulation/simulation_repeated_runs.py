"""
Run a simulation to generate point cloud from 3 orthogonal planes with different noise levels
Estimate the normals for each point and plot the results
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from normal_estimation_pca_graphs import calculate_normals_pca_graphs
from matplotlib.pyplot import cm


def create_ground_truth_point_cloud():
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    z = np.arange(0, 1, 0.1)
    xv, yv = np.meshgrid(x, y)
    points_xy_uncorrupted = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1), np.zeros_like(xv).reshape(-1, 1)))
    yv, zv = np.meshgrid(y, z)
    points_yz_uncorrupted = np.hstack((np.zeros_like(xv).reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)))
    xv, zv = np.meshgrid(x, z)
    points_xz_uncorrupted = np.hstack((xv.reshape(-1, 1), np.zeros_like(xv).reshape(-1, 1), zv.reshape(-1, 1)))
    points_gt = np.vstack((points_xy_uncorrupted, points_xz_uncorrupted, points_yz_uncorrupted))
    return points_gt


def add_noise_point_cloud(noise, points_uncorrupted):
    num_points_each_plane = int(points_uncorrupted.shape[0]/3)
    points_xy_uncorrupted = points_uncorrupted[0:num_points_each_plane, :]
    points_xz_uncorrupted = points_uncorrupted[num_points_each_plane:2*num_points_each_plane, :]
    points_yz_uncorrupted = points_uncorrupted[2*num_points_each_plane:3*num_points_each_plane, :]
    points_xy_corrupted = points_xy_uncorrupted + noise * np.random.randn(points_xy_uncorrupted.shape[0],
                                                                          points_xy_uncorrupted.shape[1])
    points_yz_corrupted = points_yz_uncorrupted + noise * np.random.randn(points_yz_uncorrupted.shape[0],
                                                                          points_yz_uncorrupted.shape[1])
    points_xz_corrupted = points_xz_uncorrupted + noise * np.random.randn(points_xz_uncorrupted.shape[0],
                                                                          points_xz_uncorrupted.shape[1])
    points_corrupted = np.vstack((points_xy_corrupted, points_xz_corrupted, points_yz_corrupted))
    return points_corrupted


# noise_levels = [0.025, 0.05, 0.075, 0.1]

num_runs = 3

points_gt = create_ground_truth_point_cloud()
alpha_levels = [0.005, 0.001, 0.005, 0.01, 0.05]
color = cm.rainbow(np.linspace(0, 1, len(alpha_levels)))
noise = 0.05
plt.rcParams.update({'font.size': 22})
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Number of iterations')
ax2.set_ylabel('Loss value [units]')
lines_loss = []

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
lines_deln = []
ax3.set_xlabel('Number of iterations')
ax3.set_ylabel('Mean(|true normal - estimated normal|) [m]')

for i, alpha in enumerate(alpha_levels):
    loss_noweight_allruns = []
    loss_weight_dotprod_allruns = []
    loss_weight_dist_allruns = []
    loss_weight_dotprod_dist_allruns = []
    deln_noweight_allruns = []
    deln_weight_dotprod_allruns = []
    deln_weight_dist_allruns = []
    deln_weight_dotprod_dist_allruns = []
    for run in range(num_runs):
        print(" Run = ", run)
        points_with_noise = add_noise_point_cloud(noise, points_uncorrupted=points_gt)
        loss, deln = calculate_normals_pca_graphs(X=points_with_noise, num_neighbors=30, alpha=alpha)
        loss_noweight_allruns.append(loss[0])
        loss_weight_dotprod_allruns.append(loss[1])
        loss_weight_dist_allruns.append(loss[2])
        loss_weight_dotprod_dist_allruns.append(loss[3])
        deln_noweight_allruns.append(deln[0])
        deln_weight_dotprod_allruns.append(deln[1])
        deln_weight_dist_allruns.append(deln[2])
        deln_weight_dotprod_dist_allruns.append(deln[3])

    length = (np.vectorize(len)(loss_noweight_allruns)).max()
    y = np.array([xi + [None] * (length - len(xi)) for xi in loss_noweight_allruns], dtype=float)
    mean_loss_noweight = np.nanmean(y, axis=0)
    std_loss_noweight = np.nanstd(y, axis=0)
    y = np.array([xi + [None] * (length - len(xi)) for xi in loss_weight_dotprod_allruns], dtype=float)
    mean_loss_weight_dotprod = np.nanmean(y, axis=0)
    std_loss_weight_dotprod = np.nanstd(y, axis=0)
    y = np.array([xi + [None] * (length - len(xi)) for xi in loss_weight_dist_allruns], dtype=float)
    mean_loss_weight_dist = np.nanmean(y, axis=0)
    std_loss_weight_dist = np.nanstd(y, axis=0)
    # length = (np.vectorize(len)(loss_weight_dotprod_dist_allruns)).max()
    y = np.array([xi + [None] * (length - len(xi)) for xi in loss_weight_dotprod_dist_allruns], dtype=float)
    mean_loss_weight_dotprod_dist = np.nanmean(y, axis=0)
    std_loss_weight_dotprod_dist = np.nanstd(y, axis=0)

    y = np.array([xi + [None] * (length - len(xi)) for xi in deln_noweight_allruns], dtype=float)
    mean_deln_noweight = np.nanmean(y, axis=0)
    std_deln_noweight = np.nanstd(y, axis=0)
    y = np.array([xi + [None] * (length - len(xi)) for xi in deln_weight_dotprod_allruns], dtype=float)
    mean_deln_weight_dotprod = np.nanmean(y, axis=0)
    std_deln_weight_dotprod = np.nanstd(y, axis=0)
    y = np.array([xi + [None] * (length - len(xi)) for xi in deln_weight_dist_allruns], dtype=float)
    mean_deln_weight_dist = np.nanmean(y, axis=0)
    std_deln_weight_dist = np.nanstd(y, axis=0)
    y = np.array([xi + [None] * (length - len(xi)) for xi in deln_weight_dotprod_dist_allruns], dtype=float)
    mean_deln_weight_dotprod_dist = np.nanmean(y, axis=0)
    std_deln_weight_dotprod_dist = np.nanstd(y, axis=0)

    plt.rcParams.update({'font.size': 22})
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    line1, = ax2.plot(range(length), mean_loss_noweight, color='tab:blue', linewidth=2, label='no weight loss')
    ax2.fill_between(range(length),
                     (mean_loss_noweight - 3*std_loss_noweight),
                     (mean_loss_noweight + 3*std_loss_noweight), color='tab:blue', alpha=.1)
    line2, = ax2.plot(range(length), mean_loss_weight_dotprod, color='tab:orange', linewidth=2, label='weight dot product loss')
    ax2.fill_between(range(length),
                     (mean_loss_weight_dotprod - 3*std_loss_weight_dotprod),
                     (mean_loss_weight_dotprod + 3*std_loss_weight_dotprod), color='tab:orange', alpha=.1)
    line3, = ax2.plot(range(length), mean_loss_weight_dist, color='tab:green', linewidth=2, label='weight dist loss')
    ax2.fill_between(range(length),
                     (mean_loss_weight_dist - 3*std_loss_weight_dist),
                     (mean_loss_weight_dist + 3*std_loss_weight_dist), color='tab:green', alpha=.1)
    line4, = ax2.plot(range(length), mean_loss_weight_dotprod_dist, color='tab:cyan', linewidth=2, label='weight dot prod dist loss')
    ax2.fill_between(range(length),
                     (mean_loss_weight_dotprod_dist - 3*std_loss_weight_dotprod_dist),
                     (mean_loss_weight_dotprod_dist + 3*std_loss_weight_dotprod_dist), color='tab:cyan', alpha=.1)
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Loss value [units]')
    ax2.legend(handles=[line1, line2, line3, line4])
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)

    line1, = ax3.plot(range(length), mean_deln_noweight, color='tab:blue', linewidth=2, label='no weight diff')
    ax2.fill_between(range(length),
                     (mean_deln_noweight - 3*std_deln_noweight),
                     (mean_deln_noweight + 3*std_deln_noweight), color='tab:blue', alpha=.1)
    line2, = ax3.plot(range(length), mean_deln_weight_dotprod, color='tab:orange', linewidth=2, label='weight dot product diff')
    ax3.fill_between(range(length),
                     (mean_deln_weight_dotprod - 3*std_deln_weight_dotprod),
                     (mean_deln_weight_dotprod + 3*std_deln_weight_dotprod), color='tab:orange', alpha=.1)
    line3, = ax3.plot(range(length), mean_deln_weight_dist, color='tab:green', linewidth=2, label='weight dist diff')
    ax3.fill_between(range(length),
                     (mean_deln_weight_dist - 3*std_deln_weight_dist),
                     (mean_deln_weight_dist + 3*std_deln_weight_dist), color='tab:green', alpha=.1)
    line4, = ax3.plot(range(length), mean_deln_weight_dotprod_dist, color='tab:cyan', linewidth=2, label='weight dot prod dist diff')
    ax3.fill_between(range(length),
                                (mean_deln_weight_dotprod_dist - 3*std_deln_weight_dotprod_dist),
                     (mean_deln_weight_dotprod_dist + 3*std_deln_weight_dotprod_dist), color='tab:cyan', alpha=.1)
    ax3.set_xlabel('Number of iterations')
    ax3.set_ylabel('Mean(|true normal - estimated normal|) [m]')
    ax3.legend()
    plt.show()

    line, = ax2.plot(range(length), mean_loss_weight_dotprod_dist, color=color[i], linewidth=2, label='alpha = ' + str(alpha))
    ax2.fill_between(range(length),
                     (mean_loss_weight_dotprod_dist - 3*std_loss_weight_dotprod_dist),
                     (mean_loss_weight_dotprod_dist + 3*std_loss_weight_dotprod_dist), color=color[i], alpha=.1)

    lines_loss.append(line)
    line, = ax3.plot(range(length), mean_deln_weight_dotprod_dist, color=color[i], linewidth=2, label='alpha = ' + str(alpha))
    ax3.fill_between(range(length),
                                (mean_deln_weight_dotprod_dist - 3*std_deln_weight_dotprod_dist),
                     (mean_deln_weight_dotprod_dist + 3*std_deln_weight_dotprod_dist), color=color[i], alpha=.1)

ax2.legend()
ax3.legend()
plt.show()