#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# internal modules
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Computes various flow statistics')
    parser.add_argument('nu', type=float, help='specify the kinematic viscosity of the flow')
    return parser.parse_args()

def main(nu):
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    #time = time[:-1]
    print("INFO: Done.", flush=True)
    print("INFO: Reading mesh...", flush=True)
    mesh = libpost.get_mesh()
    print("INFO: Done.", flush=True)
    print("INFO: Mesh sorting...", flush=True)
    mesh_pos_0 = libpost.get_object_properties(mesh, [".*__pos_0"])
    nx = int(round(mesh_pos_0["value"].size**(1/3)))
    sorting_indexs = np.empty(mesh_pos_0["value"].size, dtype=int)
    for i in range(len(mesh_pos_0["info"])):
        index = int(mesh_pos_0["info"][i].split("__")[1].split("_")[1])
        sorting_indexs[i] = index
    sorting_indexs = np.argsort(sorting_indexs)
    mesh_pos_0["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0))
    mesh =np.stack([
        mesh_pos_0["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0)), 
        libpost.get_object_properties(mesh, [".*__pos_1"])["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0)),
        libpost.get_object_properties(mesh, [".*__pos_2"])["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0))
    ])
    dx = (mesh[1, 0, 1, 0] - mesh[0, 0, 0, 0])
    print("d: ", dx * nx)
    print("INFO: Done.", flush=True)
    print("INFO: Computing flow statistics...", flush=True)
    dissipation_rate = np.empty(time.size)
    rms_flow_velocity = np.empty(time.size)
    average_streching = np.empty(time.size)
    average_vorticity = np.empty(time.size)
    integral_length_scale = np.empty(time.size)
    taylor_micro_scale_spectrum = np.empty(time.size)
    for index in range(time.size):
        t = time[index]
        print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
        # read flow
        flow = libpost.get_flow(time_dirs[time_list.index(t)])
        print(np.array(libpost.get_object_properties(flow, [".*__u_0"])["info"], dtype=str)[sorting_indexs])
        # extract velocity
        flow_velocity = np.stack([
            libpost.get_object_properties(flow, [".*__u_0"])["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0)),
            libpost.get_object_properties(flow, [".*__u_1"])["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0)),
            libpost.get_object_properties(flow, [".*__u_2"])["value"][sorting_indexs].reshape((nx, nx, nx)).transpose((2, 1, 0))
        ])
        # extract gradients
        flow_gradients = np.empty((nx**3, 3, 3))
        flow_gradients[:, 0, 0] = libpost.get_object_properties(flow, [".*__j_0_0"])["value"]
        flow_gradients[:, 0, 1] = libpost.get_object_properties(flow, [".*__j_0_1"])["value"]
        flow_gradients[:, 0, 2] = libpost.get_object_properties(flow, [".*__j_0_2"])["value"]
        flow_gradients[:, 1, 0] = libpost.get_object_properties(flow, [".*__j_1_0"])["value"]
        flow_gradients[:, 1, 1] = libpost.get_object_properties(flow, [".*__j_1_1"])["value"]
        flow_gradients[:, 1, 2] = libpost.get_object_properties(flow, [".*__j_1_2"])["value"]
        flow_gradients[:, 2, 0] = libpost.get_object_properties(flow, [".*__j_2_0"])["value"]
        flow_gradients[:, 2, 1] = libpost.get_object_properties(flow, [".*__j_2_1"])["value"]
        flow_gradients[:, 2, 2] = libpost.get_object_properties(flow, [".*__j_2_2"])["value"]
        # more gradients
        sym_flow_gradients = 0.5 * (flow_gradients + flow_gradients.transpose((0, 2, 1)))
        skew_flow_gradients = 0.5 * (flow_gradients - flow_gradients.transpose((0, 2, 1)))
        # vorticity
        vorticity = np.empty((skew_flow_gradients.shape[0], 3))
        vorticity[:, 0] = 2.0 * -skew_flow_gradients[:, 1, 2]
        vorticity[:, 1] = 2.0 * skew_flow_gradients[:, 0, 2]
        vorticity[:, 2] = 2.0 * -skew_flow_gradients[:, 0, 1]
        # compute statistics
        dissipation_rate[index] = 2.0 * nu * np.average(np.sum(np.square(sym_flow_gradients), axis=(1, 2)))
        average_flow_velocity = np.average(flow_velocity, axis=(1,2,3))
        # rms_flow_velocity[index] = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.sum(np.square(flow_velocity - average_flow_velocity.reshape((3, 1, 1, 1))), axis=0)))
        rms_flow_velocity[index] = np.std(flow_velocity)
        average_streching[index] = np.average(np.linalg.norm(sym_flow_gradients, axis=(1, 2)))
        average_vorticity[index] = np.average(np.linalg.norm(vorticity, axis=1))
        ## integral scales
        print("u.shape ", flow_velocity.shape)
        uk = np.fft.rfft(flow_velocity, axis=1) / flow_velocity.shape[1]
        print("uk.shape ", uk.shape)
        ek = 0.5 * np.average(np.sum(np.abs(uk)**2, axis=0), axis=(1,2))
        print("ek.shape ", ek.shape)
        kx = (2.0 * np.pi / (dx * nx)) * np.linspace(1, uk.shape[1] + 1, num=uk.shape[1])
        print("k = ", kx)
        print("ek = ", ek)
        integral_length_scale[index] = (np.pi / (2.0 * rms_flow_velocity[index]**2)) * np.trapz(ek / kx, x=kx)
        # tests
        taylor_micro_scale_spectrum[index] = (np.trapz(ek * kx**2, x=kx) / np.trapz(ek, x=kx))**(-0.5)
        # print
        print("\tINFO: Done.", flush=True)
    dissipation_rate = np.average(dissipation_rate)
    rms_flow_velocity = np.average(rms_flow_velocity)
    average_streching = np.average(average_streching)
    average_vorticity = np.average(average_vorticity)
    taylor_micro_scale = np.sqrt(15 * nu * np.square(rms_flow_velocity) / dissipation_rate)
    taylor_scale_reynolds = rms_flow_velocity * taylor_micro_scale / nu
    kolmogorov_time_scale = np.sqrt(nu / dissipation_rate)
    kolmogorov_length_scale = nu**(3.0/4.0) * dissipation_rate**(-1.0/4.0)
    integral_length_scale = np.average(integral_length_scale)
    print("u_rms = ", rms_flow_velocity)
    print("L = ", integral_length_scale)
    large_eddy_turnover_time = integral_length_scale / rms_flow_velocity
    # tests
    taylor_micro_scale_spectrum = np.average(taylor_micro_scale_spectrum)
    print(taylor_micro_scale_spectrum, rms_flow_velocity * taylor_micro_scale_spectrum / nu)
    print("INFO: Done.", flush=True)
    # save snapshot
    print("INFO: Saving...", flush=True)
    np.savetxt("flow_statistics.csv", np.column_stack([dissipation_rate, rms_flow_velocity, average_streching, average_vorticity, taylor_micro_scale, taylor_scale_reynolds, kolmogorov_time_scale, kolmogorov_length_scale, integral_length_scale, large_eddy_turnover_time]), delimiter=",", header="dissipation_rate,rms_flow_velocity,average_streching,average_vorticity,taylor_micro_scale,taylor_scale_reynolds,kolmogorov_time_scale,kolmogorov_length_scale,integral_length_scale,large_eddy_turnover_time")
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.nu)

# def main(nu):
    # # input
    # mesh = libpost.get_mesh()
    # flow_u_0 = libpost.get_flow_properties(["u_0"])
    # flow_u_1 = libpost.get_flow_properties(["u_1"])
    # flow_u_2 = libpost.get_flow_properties(["u_2"])
    # flow_j_0_0 = libpost.get_flow_properties(["j_0_0"])
    # flow_j_0_1 = libpost.get_flow_properties(["j_0_1"])
    # flow_j_0_2 = libpost.get_flow_properties(["j_0_2"])
    # flow_j_1_0 = libpost.get_flow_properties(["j_1_0"])
    # flow_j_1_1 = libpost.get_flow_properties(["j_1_1"])
    # flow_j_1_2 = libpost.get_flow_properties(["j_1_2"])
    # flow_j_2_0 = libpost.get_flow_properties(["j_2_0"])
    # flow_j_2_1 = libpost.get_flow_properties(["j_2_1"])
    # flow_j_2_2 = libpost.get_flow_properties(["j_2_2"])
    # # prop
    # nx = int(round((mesh["value"].shape[1]//3)**(1.0/3.0)))
    # nt = flow_u_0["value"].shape[0]
    # mesh = mesh["value"].reshape((3, nx, nx, nx))
    # u_0 = flow_u_0["value"].reshape((nt, nx, nx, nx))
    # u_1 = flow_u_1["value"].reshape((nt, nx, nx, nx))
    # u_2 = flow_u_2["value"].reshape((nt, nx, nx, nx))
    # dx = (mesh[0, 1, 0, 0] - mesh[0, 0, 0, 0])
    # # velocities
    # u_mean = np.array([np.average(flow_u_0["value"]), np.average(flow_u_1["value"]), np.average(flow_u_2["value"])]) # <- should be 0
    # u_rms = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.square(flow_u_0["value"] - u_mean[0]) + np.square(flow_u_1["value"] - u_mean[1]) + np.square(flow_u_2["value"] - u_mean[2])))
    # # turbulence flow velocity gradients properties
    # dissipation_rate = copy.deepcopy(flow_j_0_0)
    # stretch = copy.deepcopy(flow_j_0_0)
    # rotation = copy.deepcopy(flow_j_0_0)
    # for i in range(0, dissipation_rate["value"].shape[0]):
        # for j in range(0, dissipation_rate["value"].shape[1]):
            # # set J
            # J = np.empty((3, 3))
            # J[0, 0] = flow_j_0_0["value"][i, j]
            # J[0, 1] = flow_j_0_1["value"][i, j]
            # J[0, 2] = flow_j_0_2["value"][i, j]
            # J[1, 0] = flow_j_1_0["value"][i, j]
            # J[1, 1] = flow_j_1_1["value"][i, j]
            # J[1, 2] = flow_j_1_2["value"][i, j]
            # J[2, 0] = flow_j_2_0["value"][i, j]
            # J[2, 1] = flow_j_2_1["value"][i, j]
            # J[2, 2] = flow_j_2_2["value"][i, j]
            # S = 0.5 * (J + J.transpose())
            # O = 0.5 * (J - J.transpose())
            # # compute
            # dissipation_rate["value"][i, j] = 2.0 * nu * np.sum(np.square(S.flatten()))
            # #dissipation_rate["value"][i, j] = 2.0 * nu * np.sum(np.square([O[0, 1], O[0, 2], O[1, 2]]))
            # stretch["value"][i, j] = np.sum(np.abs(np.linalg.eigh(S)[0])) / 4.0
            # rotation["value"][i, j] = np.linalg.norm(np.array([-O[1, 2], O[0, 2], -O[0, 1]]))
    # dissipation_rate = np.average(dissipation_rate["value"])
    # stretch = np.average(stretch["value"])
    # rotation = np.average(rotation["value"])
    # # Taylor Micro. Scale
    # taylor_micro_scale = np.sqrt(15 * nu * np.square(u_rms) / dissipation_rate)
    # taylor_scale_reynolds = u_rms * taylor_micro_scale / nu
    # # Kolmogorov scales
    # tau_eta = np.sqrt(nu / dissipation_rate)
    # eta = nu**(3.0/4.0) * dissipation_rate**(-1.0/4.0)
    # u_eta = eta / tau_eta
    # # Integral scales
    # k = 2 * np.pi / (dx * nx) * np.linspace(1, int(nx/2 + 1)+1, num=int(nx/2 + 1))
    # integral_length_scale = np.pi / (2 * u_rms**2) * np.trapz(0.5 * np.average(np.abs(np.fft.rfft(u_0, axis=1) / u_0.shape[1])**2 + np.abs(np.fft.rfft(u_1, axis=1) / u_1.shape[1])**2 + np.abs(np.fft.rfft(u_2, axis=1) / u_2.shape[1])**2, axis=(0, 2, 3)) / k, x=k)
    # large_eddy_turnover_time = integral_length_scale / u_rms
    # # output
    # np.savetxt("flow_statistics.csv", np.array([u_rms, dissipation_rate, taylor_micro_scale, taylor_scale_reynolds, eta, tau_eta, u_eta, integral_length_scale, large_eddy_turnover_time, stretch, rotation]).reshape(1, 11), header="Root Mean Square Velocity, Turbulence Dissipation Rate, Taylor Micro. Scale, Taylor Scale Reynolds, Kolmogorov Length Scale, Kolmogorov Time Scale, Kolmogorov Velocity Scale, Integral Length Scale, Large Eddy Turnover Time, Stretch (lambda), Rotation (omega)", delimiter=",")


