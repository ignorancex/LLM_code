#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
import scipy as sp
import scipy.integrate
# test
import matplotlib.pyplot as plt
# internal modules
import libpost

BIN_RANGE = (-40.0, 40.0)#(-10.0, 10.0)
BIN_NB = 128#64#'auto'

def parse():
    parser = argparse.ArgumentParser(description='Computes statistics of the lagrangian gradients matrix (computed along particle trajectories)')
    return parser.parse_args()

def main_gradients():
    print("INFO: Post processing flow velocity gradients statistics sampled by lagrangian objects.", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names), flush=True)
    # get gradient matrix
    print("INFO: Reading objects properties...", flush=True)
    time = libpost.get_time();
    objects_j_0_0 = libpost.get_objects_properties(["j_0_0"], object_names)
    print("INFO: Done.", flush=True)
    print("INFO: Computing f(t) using pdfs...", flush=True)
    average_f = {object_name:{"value":np.empty((objects_j_0_0[object_name]["value"].shape[0], 2)), "info":["f", "95CLI"]} for object_name in objects_j_0_0}
    average_integral_f = {object_name:{"value":np.empty((objects_j_0_0[object_name]["value"].shape[0], 2)), "info":["integral_f", "95CLI"]} for object_name in objects_j_0_0}
    average_tau = {object_name:{"value":np.empty(2), "info":["tau", "95CLI"]} for object_name in objects_j_0_0}
    fig = plt.figure() # PLT
    fig.add_axes(plt.gca()) # PLT
    ax = fig.get_axes()[0] # PLT
    ax.set_aspect('equal', 'box') # PLT
    for object_name in objects_j_0_0:
        print("INFO:    Processing " + object_name + "...", flush=True)
        # init
        average_f[object_name]["value"][0, 0] = 1.0
        average_f[object_name]["value"][0, 1] = 0.0
        average_integral_f[object_name]["value"][0, 0] = 0.0
        average_integral_f[object_name]["value"][0, 1] = 0.0
        for k in range(1, objects_j_0_0[object_name]["value"].shape[0]):
            j_t = objects_j_0_0[object_name]["value"][:-k, :].flatten()
            j_0 = objects_j_0_0[object_name]["value"][k:, :].flatten()
            pdf, edges = np.histogramdd((
                j_0,
                j_t
            ), bins=(np.histogram_bin_edges(j_0, bins=BIN_NB, range=BIN_RANGE), np.histogram_bin_edges(j_t, bins=BIN_NB, range=BIN_RANGE)), density=True)
            if k < 64: # PLT
                ax.clear() # PLT
                ax.pcolormesh(edges[0], edges[1], pdf) # PLT
                fig.savefig("pdf_{0}_{1:03d}.png".format(object_name, k)) # PLT
            # compute f
            ## remove zeros
            sum__j_t__p_j_t_j_0 = np.sum(pdf * 0.5 * (edges[1][1:] + edges[1][:-1]) * np.diff(edges[1]), axis=1)
            j_0_p_j_0 = np.sum(pdf * np.diff(edges[1]), axis=1) * 0.5 * (edges[0][1:] + edges[0][:-1])
            mask = (j_0_p_j_0 != 0.0)
            f_value = sum__j_t__p_j_t_j_0[mask] / j_0_p_j_0[mask]
            ## computation
            average_f[object_name]["value"][k, 0] = np.average(f_value)
            average_f[object_name]["value"][k, 1] = 1.96 * np.std(f_value) / np.sqrt(f_value.size)
        # compute integral f
        average_integral_f[object_name]["value"][0, 0] = 0.0
        average_integral_f[object_name]["value"][1:, 0] = sp.integrate.cumtrapz(average_f[object_name]["value"][:, 0], x=time, axis=0)
        average_integral_f[object_name]["value"][:, 1] = -1.0
        # compute tau
        average_tau[object_name]["value"][0] = np.trapz(average_integral_f[object_name]["value"][:, 0], x=time, axis=0) / time[-1]
        average_tau[object_name]["value"][1] = -1.0
        print("INFO:    " + object_name + " done.", flush=True)
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving...", flush=True)
    libpost.savet(time, average_f, "f_0_0")
    libpost.savet(time, average_integral_f, "integral_f_0_0")
    libpost.save(average_tau, "tau_f_0_0")
    print("INFO: Done.", flush=True)

def main_velocity():
    print("INFO: Post processing flow velocity statistics sampled by lagrangian objects.", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names), flush=True)
    # get gradient matrix
    print("INFO: Reading objects properties...", flush=True)
    time = libpost.get_time();
    # gradients
    objects_j_0_0 = libpost.get_objects_properties(["j_0_0"], object_names)
    # objects_j_0_1 = libpost.get_objects_properties(["j_0_1"], object_names)
    # objects_j_0_2 = libpost.get_objects_properties(["j_0_2"], object_names)
    # objects_j_1_0 = libpost.get_objects_properties(["j_1_0"], object_names)
    # objects_j_1_1 = libpost.get_objects_properties(["j_1_1"], object_names)
    # objects_j_1_2 = libpost.get_objects_properties(["j_1_2"], object_names)
    # objects_j_2_0 = libpost.get_objects_properties(["j_2_0"], object_names)
    # objects_j_2_1 = libpost.get_objects_properties(["j_2_1"], object_names)
    # objects_j_2_2 = libpost.get_objects_properties(["j_2_2"], object_names)
    # velocity
    objects_u_0 = libpost.get_objects_properties(["u_0"], object_names)
    # objects_u_1 = libpost.get_objects_properties(["u_1"], object_names)
    # objects_u_2 = libpost.get_objects_properties(["u_2"], object_names)
    # position
    objects_pos_0 = libpost.get_objects_properties(["pos_0"], object_names)
    # objects_pos_1 = libpost.get_objects_properties(["pos_1"], object_names)
    # objects_pos_2 = libpost.get_objects_properties(["pos_2"], object_names)
    print("INFO: Done.", flush=True)
    # print("INFO: Reconstructing gradients, velocity and position.", flush=True)
    # gradients_value = {}
    # velocity_value = {}
    # position_value = {}
    # for object_name in objects_j_0_0:
        # # gradients
        # gradients_value[object_name] = np.empty((objects_j_0_0[object_name]["value"].shape[0], objects_j_0_0[object_name]["value"].shape[1], 3, 3))
        # gradients_value[object_name][:, :, 0, 0] = objects_j_0_0[object_name]["value"]
        # gradients_value[object_name][:, :, 0, 1] = objects_j_0_1[object_name]["value"]
        # gradients_value[object_name][:, :, 0, 2] = objects_j_0_2[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 0] = objects_j_1_0[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 1] = objects_j_1_1[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 2] = objects_j_1_2[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 0] = objects_j_2_0[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 1] = objects_j_2_1[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 2] = objects_j_2_2[object_name]["value"]
        # # velocity
        # velocity_value[object_name] = np.empty((objects_u_0[object_name]["value"].shape[0], objects_u_0[object_name]["value"].shape[1], 3))
        # velocity_value[object_name][:, :, 0] = np.empty((objects_u_0[object_name]["value"].shape[0], objects_u_0[object_name]["value"].shape[1], 3))
        # velocity_value[object_name][:, :, 1] = np.empty((objects_u_1[object_name]["value"].shape[0], objects_u_1[object_name]["value"].shape[1], 3))
        # velocity_value[object_name][:, :, 2] = np.empty((objects_u_2[object_name]["value"].shape[0], objects_u_2[object_name]["value"].shape[1], 3))
        # # position
        # position_value[object_name] = np.empty((objects_pos_0[object_name]["value"].shape[0], objects_pos_0[object_name]["value"].shape[1], 3))
        # position_value[object_name][:, :, 0] = np.empty((objects_pos_0[object_name]["value"].shape[0], objects_pos_0[object_name]["value"].shape[1], 3))
        # position_value[object_name][:, :, 1] = np.empty((objects_pos_1[object_name]["value"].shape[0], objects_pos_1[object_name]["value"].shape[1], 3))
        # position_value[object_name][:, :, 2] = np.empty((objects_pos_2[object_name]["value"].shape[0], objects_pos_2[object_name]["value"].shape[1], 3))
    # print("INFO: Done.", flush=True)
    print("INFO: Computing f(t) using pdfs...", flush=True)
    average_f = {object_name:{"value":np.empty((objects_j_0_0[object_name]["value"].shape[0], 2)), "info":["f", "95CLI"]} for object_name in objects_j_0_0}
    average_integral_f = {object_name:{"value":np.empty((objects_j_0_0[object_name]["value"].shape[0], 2)), "info":["integral_f", "95CLI"]} for object_name in objects_j_0_0}
    average_tau = {object_name:{"value":np.empty(2), "info":["tau", "95CLI"]} for object_name in objects_j_0_0}
    for object_name in objects_j_0_0:
        print("INFO:    Processing " + object_name + "...", flush=True)
        # init
        average_f[object_name]["value"][0, 0] = 1.0
        average_f[object_name]["value"][0, 1] = 0.0
        average_integral_f[object_name]["value"][0, 0] = 0.0
        average_integral_f[object_name]["value"][0, 1] = 0.0
        for k in range(1, objects_j_0_0[object_name]["value"].shape[0]):
            u_t = np.concatenate((objects_u_0[object_name]["value"][:-k, :].flatten(), objects_u_0[object_name]["value"][k:, :].flatten()))
            x_t = np.concatenate((objects_pos_0[object_name]["value"][:-k, :].flatten(), objects_pos_0[object_name]["value"][k:, :].flatten()))
            j_0 = np.concatenate((objects_j_0_0[object_name]["value"][k:, :].flatten(), objects_j_0_0[object_name]["value"][:-k, :].flatten()))
            pdf_j_u, edges_j_u = np.histogramdd((
                j_0,
                u_t
            ), bins=(np.histogram_bin_edges(j_0, bins=BIN_NB), np.histogram_bin_edges(u_t, bins=BIN_NB)), range=((j_0.min(), j_0.max()), (u_t.min(), u_t.max())), density=True)
            pdf_j_x, edges_j_x = np.histogramdd((
                j_0,
                x_t
            ), bins=(np.histogram_bin_edges(j_0, bins=BIN_NB), np.histogram_bin_edges(x_t, bins=BIN_NB)), range=((j_0.min(), j_0.max()), (x_t.min(), x_t.max())), density=True)
            # compute f
            ## remove zeros
            sum_u_t_p_j_0_u_t = np.sum(pdf_j_u * 0.5 * (edges_j_u[1][1:] + edges_j_u[1][:-1]) * np.diff(edges_j_u[1]), axis=1)
            j_0__sum_j_t_p_j_0_x_t = 0.5 * (edges_j_u[0][1:] + edges_j_u[0][:-1]) * np.sum(pdf_j_x * 0.5 * (edges_j_x[1][1:] + edges_j_x[1][:-1]) * np.diff(edges_j_x[1]), axis=1)
            mask = (j_0__sum_j_t_p_j_0_x_t != 0.0)
            f_value = sum_u_t_p_j_0_u_t[mask] / j_0__sum_j_t_p_j_0_x_t[mask]
            ## computation
            average_f[object_name]["value"][k, 0] = np.average(f_value)
            average_f[object_name]["value"][k, 1] = 1.96 * np.std(f_value) / np.sqrt(f_value.size)
        # compute integral f
        average_integral_f[object_name]["value"][0, 0] = 0.0
        average_integral_f[object_name]["value"][1:, 0] = sp.integrate.cumtrapz(average_f[object_name]["value"][:, 0], x=time, axis=0)
        average_integral_f[object_name]["value"][:, 1] = -1.0
        # compute tau
        average_tau[object_name]["value"][0] = np.trapz(average_integral_f[object_name]["value"][:, 0], x=time, axis=0) / time[-1]
        average_tau[object_name]["value"][1] = -1.0
        print("INFO:    " + object_name + " done.", flush=True)
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving...", flush=True)
    libpost.savet(time, average_f, "f_0_0")
    libpost.savet(time, average_integral_f, "integral_f_0_0")
    libpost.save(average_tau, "tau_f_0_0")
    print("INFO: Done.", flush=True)

# def main_save():
    # print("INFO: Post processing flow velocity gradients statistics sampled by lagrangian objects.")
    # object_names = libpost.get_object_names()
    # print("INFO: Object names are:", " ".join(object_names)[:-1])
    # # get gradient matrix
    # print("INFO: Reading objects properties...")
    # time = libpost.get_time();
    # objects_j_0_0 = libpost.get_objects_properties(["j_0_0"], object_names)
    # objects_j_0_1 = libpost.get_objects_properties(["j_0_1"], object_names)
    # objects_j_0_2 = libpost.get_objects_properties(["j_0_2"], object_names)
    # objects_j_1_0 = libpost.get_objects_properties(["j_1_0"], object_names)
    # objects_j_1_1 = libpost.get_objects_properties(["j_1_1"], object_names)
    # objects_j_1_2 = libpost.get_objects_properties(["j_1_2"], object_names)
    # objects_j_2_0 = libpost.get_objects_properties(["j_2_0"], object_names)
    # objects_j_2_1 = libpost.get_objects_properties(["j_2_1"], object_names)
    # objects_j_2_2 = libpost.get_objects_properties(["j_2_2"], object_names)
    # print("INFO: Done.")
    # print("INFO: Gradients reconstruction...")
    # gradients_value = {}
    # sym_gradients_value = {}
    # skew_gradients_value = {}
    # for object_name in objects_j_0_0:
        # gradients_value[object_name] = np.empty((objects_j_0_0[object_name]["value"].shape[0], objects_j_0_0[object_name]["value"].shape[1], 3, 3))
        # gradients_value[object_name][:, :, 0, 0] = objects_j_0_0[object_name]["value"]
        # gradients_value[object_name][:, :, 0, 1] = objects_j_0_1[object_name]["value"]
        # gradients_value[object_name][:, :, 0, 2] = objects_j_0_2[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 0] = objects_j_1_0[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 1] = objects_j_1_1[object_name]["value"]
        # gradients_value[object_name][:, :, 1, 2] = objects_j_1_2[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 0] = objects_j_2_0[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 1] = objects_j_2_1[object_name]["value"]
        # gradients_value[object_name][:, :, 2, 2] = objects_j_2_2[object_name]["value"]
        # sym_gradients_value[object_name] = 0.5 * (gradients_value[object_name] + gradients_value[object_name].transpose(0, 1, 3, 2))
        # skew_gradients_value[object_name] = 0.5 * (gradients_value[object_name] - gradients_value[object_name].transpose(0, 1, 3, 2))
    # print("INFO: Done.")
    # print("INFO: Cleaning...")
    # del objects_j_0_0
    # del objects_j_0_1
    # del objects_j_0_2
    # del objects_j_1_0
    # del objects_j_1_1
    # del objects_j_1_2
    # del objects_j_2_0
    # del objects_j_2_1
    # del objects_j_2_2
    # print("INFO: Done.")
    # print("INFO: Saving flow gradients statistics...")
    # statistics = {}
    # for object_name in gradients_value:
        # gradients_norm = np.linalg.norm(gradients_value[object_name], axis=(2, 3))
        # skew_gradients_norm = np.linalg.norm(skew_gradients_value[object_name], axis=(2, 3))
        # sym_gradients_norm = np.linalg.norm(sym_gradients_value[object_name], axis=(2, 3))
        # statistics[object_name] = np.array([
            # # gradients
            # np.average(gradients_norm),
            # 1.96 * np.std(gradients_norm) / np.sqrt(gradients_value[object_name].shape[0] * gradients_value[object_name].shape[1]),
            # # sym gradients
            # np.average(sym_gradients_norm),
            # 1.96 * np.std(sym_gradients_norm) / np.sqrt(sym_gradients_value[object_name].shape[0] * sym_gradients_value[object_name].shape[1]),
            # # skew gradients
            # np.average(skew_gradients_norm),
            # 1.96 * np.std(skew_gradients_norm) / np.sqrt(skew_gradients_value[object_name].shape[0] * skew_gradients_value[object_name].shape[1]),
        # ]).reshape(1, 6)
        # np.savetxt("flow_gradients_statistics__"+object_name+".csv", statistics[object_name], header="E[|A|], 95CLI, E[|S|], 95CLI, E[|O|], 95CLI", delimiter=",")
    # print("INFO: Done.")
    # print("INFO: Computing gradients error...")
    # objects_gradients_error_2 = {object_name:{"value":np.empty((gradients_value[object_name].shape[0], 3)), "info":["E[|A_t - A_0|^2]", "sum_V[a_t - a_0]", "f"]} for object_name in gradients_value}
    # objects_skew_gradients_error_2 = {object_name:{"value":np.empty((gradients_value[object_name].shape[0], 3)), "info":["E[|S_t - S_0|^2]", "sum_V[s_t - s_0]", "f"]} for object_name in gradients_value}
    # objects_sym_gradients_error_2 = {object_name:{"value":np.empty((gradients_value[object_name].shape[0], 3)), "info":["E[|O_t - O_0|^2]", "sum_V[o_t - o_0]", "f"]} for object_name in gradients_value}
    # for object_name in gradients_value:
        # print("INFO:    Processing " + object_name + "...")
        # # value
        # objects_gradients_error_2[object_name]["value"][0, 0] = 0.0
        # objects_gradients_error_2[object_name]["value"][0, 1] = 0.0
        # objects_gradients_error_2[object_name]["value"][0, 2] = 1.0
        # objects_skew_gradients_error_2[object_name]["value"][0, 0] = 0.0
        # objects_skew_gradients_error_2[object_name]["value"][0, 1] = 0.0
        # objects_skew_gradients_error_2[object_name]["value"][0, 2] = 1.0
        # objects_sym_gradients_error_2[object_name]["value"][0, 0] = 0.0
        # objects_sym_gradients_error_2[object_name]["value"][0, 1] = 0.0
        # objects_sym_gradients_error_2[object_name]["value"][0, 2] = 1.0
        # for k in range(1, gradients_value[object_name].shape[0]):
            # # gradients error
            # norm = np.array([
                # np.linalg.norm(gradients_value[object_name][:-k, :, :, :] - gradients_value[object_name][k:, :, :, :], axis=(2, 3))**2,
                # np.linalg.norm(gradients_value[object_name][k:, :, :, :] - gradients_value[object_name][:-k, :, :, :], axis=(2, 3))**2
            # ])
            # trace = np.sum(np.var(np.concatenate(
                # (gradients_value[object_name][:-k, :, :, :] - gradients_value[object_name][k:, :, :, :],
                # gradients_value[object_name][k:, :, :, :] - gradients_value[object_name][:-k, :, :, :]),
                # axis=1
            # ), axis=(0, 1)))
            # objects_gradients_error_2[object_name]["value"][k, 0] = np.average(norm)
            # objects_gradients_error_2[object_name]["value"][k, 1] = trace
            # objects_gradients_error_2[object_name]["value"][k, 2] = 1 - np.sqrt((objects_gradients_error_2[object_name]["value"][k, 0] - objects_gradients_error_2[object_name]["value"][k, 1]) / statistics[object_name][0, 2])
            # # skew gradients error
            # norm = np.array([
                # np.linalg.norm(skew_gradients_value[object_name][:-k, :, :, :] - skew_gradients_value[object_name][k:, :, :, :], axis=(2, 3))**2,
                # np.linalg.norm(skew_gradients_value[object_name][k:, :, :, :] - skew_gradients_value[object_name][:-k, :, :, :], axis=(2, 3))**2
            # ])
            # trace = np.sum(np.var(np.concatenate(
                # (skew_gradients_value[object_name][:-k, :, :, :] - skew_gradients_value[object_name][k:, :, :, :],
                # skew_gradients_value[object_name][k:, :, :, :] - skew_gradients_value[object_name][:-k, :, :, :]),
                # axis=1
            # ), axis=(0, 1)))
            # objects_skew_gradients_error_2[object_name]["value"][k, 0] = np.average(norm)
            # objects_skew_gradients_error_2[object_name]["value"][k, 1] = trace
            # objects_skew_gradients_error_2[object_name]["value"][k, 2] = 1 - np.sqrt((objects_skew_gradients_error_2[object_name]["value"][k, 0] - objects_skew_gradients_error_2[object_name]["value"][k, 1]) / statistics[object_name][0, 10])
            # # sym gradients error
            # norm = np.array([
                # np.linalg.norm(sym_gradients_value[object_name][:-k, :, :, :] - sym_gradients_value[object_name][k:, :, :, :], axis=(2, 3))**2,
                # np.linalg.norm(sym_gradients_value[object_name][k:, :, :, :] - sym_gradients_value[object_name][:-k, :, :, :], axis=(2, 3))**2
            # ])
            # trace = np.sum(np.var(np.concatenate(
                # (sym_gradients_value[object_name][:-k, :, :, :] - sym_gradients_value[object_name][k:, :, :, :],
                # sym_gradients_value[object_name][k:, :, :, :] - sym_gradients_value[object_name][:-k, :, :, :]),
                # axis=1
            # ), axis=(0, 1)))
            # objects_sym_gradients_error_2[object_name]["value"][k, 0] = np.average(norm)
            # objects_sym_gradients_error_2[object_name]["value"][k, 1] = trace
            # objects_sym_gradients_error_2[object_name]["value"][k, 2] = 1 - np.sqrt((objects_sym_gradients_error_2[object_name]["value"][k, 0] - objects_sym_gradients_error_2[object_name]["value"][k, 1]) / statistics[object_name][0, 6])
        # print("INFO:    " + object_name + " done.")
    # print("INFO: Done.")
    # # save
    # print("INFO: Saving...")
    # libpost.savet(time, objects_gradients_error_2, "average_gradients_error_2")
    # libpost.savet(time, objects_skew_gradients_error_2, "average_skew_gradients_error_2")
    # libpost.savet(time, objects_sym_gradients_error_2, "average_sym_gradients_error_2")
    # print("INFO: Done.")

if __name__ == '__main__':
    args = parse()
    main_gradients()
