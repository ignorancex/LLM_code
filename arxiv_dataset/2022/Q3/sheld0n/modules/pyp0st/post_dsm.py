#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# file
import os.path
# numpy
import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg
import scipy.interpolate
# internal modules
import libpost
# tmp plot
import matplotlib.pyplot as plt

# nu: 0.000185
# epsilon: 0.103
# kolmogorov length scale: 0.0028
# kolmogorov time scale: 0.0424
# batchelor scale: (nu D^2 / epsilon)^(1/4)
# batchelor (scale < kolmogorov length) eq (D < nu)
# D < 1.85e-4

sc = 1e0
D = 1.85e-4 / sc
s0 = 1e0 * 0.0028 # initial thickness
dA0 = (1e0 * 0.0028)**2 # initial surface

c_nb = 100
c_array = np.linspace(0.0, 1.0, num=c_nb)
C_nb = c_nb
C_array = np.linspace(0.0, 1.0, num=C_nb)
rho_nb = 10 * c_nb
log_rho_array = np.linspace(-np.log(10), np.log(100000), num=rho_nb)

#process_times = [0.0, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
process_times = [0.002, 0.004, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 1.536, 2.048, 4.096, 8.192] # step 0.002

def parse():
    parser = argparse.ArgumentParser(description='Computes statistics of the lagrangian gradients matrix (computed along particle trajectories)')
    return parser.parse_args()

# computing lyapunov exponent

def rho_orientation(theta, phi, mu_pp, mu_p, mu):
    return (np.sin(theta)**2 * (mu_pp * np.cos(phi)**2 + mu_p * np.sin(phi)**2) + mu * np.cos(theta)**2)**0.5

def compute_lyapunov_exponent(mu_pp, mu_p, mu, t):
    theta = np.linspace(0.0, np.pi/2.0, num=100)
    phi = np.linspace(0.0, np.pi/2.0, num=100)
    mu_pp = np.tile(mu_pp, (100, 100, 1)).transpose((2, 0, 1))
    mu_p = np.tile(mu_p, (100, 100, 1)).transpose((2, 0, 1))
    mu = np.tile(mu, (100, 100, 1)).transpose((2, 0, 1))
    THETA, PHI = np.meshgrid(theta, phi)
    return np.trapz(np.trapz(np.log(rho_orientation(THETA, PHI, mu_pp, mu_p, mu))/t * np.sin(THETA)/(np.pi/2.0), x=theta, axis=2), x=phi, axis=1)

# computing concentration

def dL_dt(foperator, t, L):
    return np.matmul(foperator(t), L.reshape((3,3), order='C')).flatten(order='C')

def F(a):
    return np.log((4.0/a + 1) + np.sqrt((4.0/a + 1)**2 - 1.0)) - 1.6/np.sqrt(a + 1.4)

def P(rho, mu_pp, mu_p, mu):
    # init result
    result = np.zeros((rho.size, mu.size))
    # broadcast
    rho = np.broadcast_to(rho.reshape((-1, 1)), result.shape)
    mu_pp = np.broadcast_to(mu_pp, result.shape)
    mu_p = np.broadcast_to(mu_p, result.shape)
    mu = np.broadcast_to(mu, result.shape)
    #mask
    mask_eq_2o37 = (np.power(rho, 2) > mu_p) & (np.power(rho, 2) < mu)
    mask_eq_2o38 = (np.power(rho, 2) > mu_pp) & (np.power(rho, 2) < mu_p)
    # compute
    result[mask_eq_2o37] = rho[mask_eq_2o37] / (np.pi * np.sqrt(mu[mask_eq_2o37] - np.power(rho[mask_eq_2o37], 2)) * np.sqrt(mu_p[mask_eq_2o37] - mu_pp[mask_eq_2o37])) * F(2 * (np.power(rho[mask_eq_2o37], 2) - mu_p[mask_eq_2o37]) * (mu[mask_eq_2o37] - mu_pp[mask_eq_2o37]) / ((mu[mask_eq_2o37] - np.power(rho[mask_eq_2o37], 2)) * (mu_p[mask_eq_2o37] - mu_pp[mask_eq_2o37])))
    result[mask_eq_2o38] = rho[mask_eq_2o38] / (np.pi * np.sqrt(mu[mask_eq_2o38] - mu_p[mask_eq_2o38]) * np.sqrt(np.power(rho[mask_eq_2o38], 2) - mu_pp[mask_eq_2o38])) * F(2 * (mu_p[mask_eq_2o38] - np.power(rho[mask_eq_2o38], 2)) * (mu[mask_eq_2o38] - mu_pp[mask_eq_2o38]) / ((np.power(rho[mask_eq_2o38], 2) - mu_pp[mask_eq_2o38]) * (mu[mask_eq_2o38] - mu_p[mask_eq_2o38])))
    # return
    return result

def Q(C, eta_pp, eta_p, eta):
    # init result
    result = np.zeros((C.size, eta.size))
    # broadcast
    C = np.broadcast_to(C.reshape((-1, 1)), result.shape)
    eta_pp = np.broadcast_to(eta_pp, result.shape)
    eta_p = np.broadcast_to(eta_p, result.shape)
    eta = np.broadcast_to(eta, result.shape)
    #mask
    mask_eq_2o43 = (np.power(C, -2) > eta_pp) & (np.power(C, -2) < eta_p)
    mask_eq_2o44 = (np.power(C, -2) > eta_p) & (np.power(C, -2) < eta)
    # compute
    result[mask_eq_2o43] = dA0 * s0 / (np.pi * np.power(C[mask_eq_2o43], 4) * np.sqrt(eta[mask_eq_2o43] - eta_p[mask_eq_2o43]) * np.sqrt(np.power(C[mask_eq_2o43], -2) - eta_pp[mask_eq_2o43])) * F(2 * (eta_p[mask_eq_2o43] - np.power(C[mask_eq_2o43], -2)) * (eta[mask_eq_2o43] - eta_pp[mask_eq_2o43]) / ((np.power(C[mask_eq_2o43], -2) - eta_pp[mask_eq_2o43]) * (eta[mask_eq_2o43] - eta_p[mask_eq_2o43])))
    result[mask_eq_2o44] = dA0 * s0 / (np.pi * np.power(C[mask_eq_2o44], 4) * np.sqrt(eta[mask_eq_2o44] - np.power(C[mask_eq_2o44], -2)) * np.sqrt(eta_p[mask_eq_2o44] - eta_pp[mask_eq_2o44])) * F(2 * (np.power(C[mask_eq_2o44], -2) - eta_p[mask_eq_2o44]) * (eta[mask_eq_2o44] - eta_pp[mask_eq_2o44]) / ((eta[mask_eq_2o44] - np.power(C[mask_eq_2o44], -2)) * (eta_p[mask_eq_2o44] - eta_pp[mask_eq_2o44])))
    # return
    return result

def main():
    print("INFO: Post processing the diffuselet method.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    #time = time[:10]
    #average_time_steps = np.round(np.linspace(0, time.size-1, n_average)).astype(int)
    print("INFO: Done.", flush=True)
    data = {}
    if not os.path.isdir("objects_npy"):
        os.mkdir("objects_npy")
    print("INFO: Reading int L^t L...", flush=True)
    int_Lt_L_to_process = []
    for object_name in object_names:
        if os.path.isfile("objects_npy/" + object_name + "_int_Lt_L.npy"):
            data[object_name] = {
                "L^t_L":np.load("objects_npy/" + object_name + "_Lt_L.npy"),
                "int_L^t_L":np.load("objects_npy/" + object_name + "_int_Lt_L.npy")
            }
        else:
            int_Lt_L_to_process.append(object_name)
    print("INFO: Done.", flush=True)
    if int_Lt_L_to_process:
        print("INFO: Reading L...", flush=True)
        L_to_process = []
        for object_name in int_Lt_L_to_process:
            if os.path.isfile("objects_npy/" + object_name + "_L.npy"):
                data[object_name] = {"L":np.load("objects_npy/" + object_name + "_L.npy")}
            else:
                L_to_process.append(object_name)
        print("INFO: Done.", flush=True)
        if L_to_process:
            print("INFO: Extracting gradients...", flush=True)
            for name in L_to_process:
                print("\tINFO: Processing object {name}...".format(name=name), flush=True)
                print("\t\tINFO: Processing t={t}/{t_f}...".format(t=0, t_f=time[-1]), flush=True)
                obj = libpost.get_object(time_dirs[time_list.index(0.0)], name)
                obj_j_0_0 = libpost.get_object_properties(obj, ["particle_.*__j_0_0"])["value"]
                data[name] = {"grads":np.empty((time.size, obj_j_0_0.size, 3, 3))}
                data[name]["grads"][0, :, 0, 0] = obj_j_0_0
                del obj_j_0_0
                data[name]["grads"][0, :, 0, 1] = libpost.get_object_properties(obj, ["particle_.*__j_0_1"])["value"]
                data[name]["grads"][0, :, 0, 2] = libpost.get_object_properties(obj, ["particle_.*__j_0_2"])["value"]
                data[name]["grads"][0, :, 1, 0] = libpost.get_object_properties(obj, ["particle_.*__j_1_0"])["value"]
                data[name]["grads"][0, :, 1, 1] = libpost.get_object_properties(obj, ["particle_.*__j_1_1"])["value"]
                data[name]["grads"][0, :, 1, 2] = libpost.get_object_properties(obj, ["particle_.*__j_1_2"])["value"]
                data[name]["grads"][0, :, 2, 0] = libpost.get_object_properties(obj, ["particle_.*__j_2_0"])["value"]
                data[name]["grads"][0, :, 2, 1] = libpost.get_object_properties(obj, ["particle_.*__j_2_1"])["value"]
                data[name]["grads"][0, :, 2, 2] = libpost.get_object_properties(obj, ["particle_.*__j_2_2"])["value"]
                print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=0, t_f=time[-1]), flush=True)
                for index in range(1, time.size):
                    t = time[index]
                    print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
                    # read object
                    obj = libpost.get_object(time_dirs[time_list.index(t)], name)
                    data[name]["grads"][index, :, 0, 0] = libpost.get_object_properties(obj, ["particle_.*__j_0_0"])["value"]
                    data[name]["grads"][index, :, 0, 1] = libpost.get_object_properties(obj, ["particle_.*__j_0_1"])["value"]
                    data[name]["grads"][index, :, 0, 2] = libpost.get_object_properties(obj, ["particle_.*__j_0_2"])["value"]
                    data[name]["grads"][index, :, 1, 0] = libpost.get_object_properties(obj, ["particle_.*__j_1_0"])["value"]
                    data[name]["grads"][index, :, 1, 1] = libpost.get_object_properties(obj, ["particle_.*__j_1_1"])["value"]
                    data[name]["grads"][index, :, 1, 2] = libpost.get_object_properties(obj, ["particle_.*__j_1_2"])["value"]
                    data[name]["grads"][index, :, 2, 0] = libpost.get_object_properties(obj, ["particle_.*__j_2_0"])["value"]
                    data[name]["grads"][index, :, 2, 1] = libpost.get_object_properties(obj, ["particle_.*__j_2_1"])["value"]
                    data[name]["grads"][index, :, 2, 2] = libpost.get_object_properties(obj, ["particle_.*__j_2_2"])["value"]
                    print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
                data[name]["operator"] = -data[name]["grads"].transpose(0, 1, 3, 2)
                del data[name]["grads"]
                print("\tINFO: Done processing object {name}.".format(name=name), flush=True)
            print("INFO: Done.", flush=True)
            print("INFO: Computing L...", flush=True)
            for object_name in L_to_process:
                data[object_name]["L"] = np.empty(data[object_name]["operator"].shape)
                print("\tINFO: Processing object {name}...".format(name=object_name), flush=True)
                for j in range(data[object_name]["operator"].shape[1]):
                    foperator = sp.interpolate.interp1d(time, data[object_name]["operator"][:, j, :, :], kind="cubic", axis=0, fill_value="extrapolate")
                    data[object_name]["L"][:, j, :, :] = sp.integrate.solve_ivp(lambda t, L : dL_dt(foperator, t, L), (0.0, time[-1]), np.identity(3).flatten(order='C'), t_eval=time).y.reshape((data[object_name]["operator"].shape[0], 3, 3), order='C')
                print("\tINFO: Done processing object {name}.".format(name=object_name), flush=True)
            print("INFO: Done.", flush=True)
            print("INFO: Saving L...", flush=True)
            for object_name in L_to_process:
                np.save("objects_npy/" + object_name + "_L", data[object_name]["L"])
            print("INFO: Done.", flush=True)
        print("INFO: Computing int L^t L...", flush=True)
        for object_name in int_Lt_L_to_process:
            print("\tINFO: Processing object {name}...".format(name=object_name), flush=True)
            # value
            data[object_name]["L^t_L"] = np.matmul(data[object_name]["L"].transpose(0, 1, 3, 2), data[object_name]["L"], axes=[(2, 3), (2, 3), (2, 3)])
            data[object_name]["int_L^t_L"] = sp.integrate.cumtrapz(data[object_name]["L^t_L"], axis=0, x=time, initial=0)
            print("\tINFO: Done processing object {name}.".format(name=object_name), flush=True)
        print("INFO: Done.", flush=True)
        print("INFO: Saving int_Lt_L...", flush=True)
        for object_name in int_Lt_L_to_process:
            np.save("objects_npy/" + object_name + "_Lt_L", data[object_name]["L^t_L"])
            np.save("objects_npy/" + object_name + "_int_Lt_L", data[object_name]["int_L^t_L"])
        print("INFO: Done.", flush=True)
    print("INFO: Computing Tau...", flush=True)
    tau = {}
    for object_name in object_names:
        print("\tINFO: Processing object {name}...".format(name=object_name), flush=True)
        # value
        tau[object_name] = np.identity(3) + 4.0 * D / s0**2 * data[object_name]["int_L^t_L"]
        print("\tINFO: Done processing object {name}.".format(name=object_name), flush=True)
    print("INFO: Done.", flush=True)
    print("INFO: Computing c pdf...", flush=True)
    for object_name in tau:
        c_var = np.empty(len(process_times))
        lyapunov_exponent = np.empty(len(process_times))
        lyapunov_exponent_2 = np.empty(len(process_times))
        log_rho_var = np.empty(len(process_times))
        print("\tINFO: Processing object {name}...".format(name=object_name), flush=True)
        for process_times_index in range(len(process_times)):
            t = process_times[process_times_index]
            time_index = time.tolist().index(t)
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=process_times[-1]), flush=True)
            # compute concentration pdf
            c_pdf = np.zeros(c_nb)
            eigvals = np.linalg.eigvalsh(tau[object_name][time_index, :, :, :])
            print(np.sqrt(1.0 / np.average(eigvals, axis=0)))
            print(np.average(eigvals, axis=0))
            for c_index in range(1, c_nb - 1):
                c = c_array[c_index]
                Q_sum = np.zeros(C_nb)
                Q_sum[c_index + 1:] = np.sum(Q(C_array[c_index + 1:], eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]), axis=1) / (c * np.sqrt(np.log(C_array[c_index + 1:]/c)))
                c_pdf[c_index] = np.trapz(Q_sum, x=C_array) / tau[object_name].shape[1]
            # save
            np.savetxt("{}__time_{}__sc_{}__c_pdf.csv".format(object_name, str(t).replace(".", "o"), str(sc).replace(".", "o")), np.column_stack((c_array, c_pdf)), delimiter=",", header="c, p(c)")
            # compute variance
            # c_var[process_times_index] = np.sqrt(0.5 * np.pi) * np.average(np.trapz(np.power(C_array, 2) * Q(C_array, eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]).transpose(), x=C_array))
            c_var[process_times_index] = np.average(np.trapz(np.power(c_array, 2) * c_pdf, x=c_array))
            # compute stretching pdf
            eigvals = np.linalg.eigvalsh(data[object_name]["L^t_L"][time_index, :, :, :])
            log_rho_pdf = np.nanmean(np.exp(log_rho_array).reshape(-1, 1) * P(np.exp(log_rho_array), eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]), axis=1)
            np.savetxt("{}__time_{}__log_rho_pdf.csv".format(object_name, str(t).replace(".", "o")), np.column_stack((log_rho_array, log_rho_pdf)), delimiter=",", header="log(rho), p(log(rho))")
            # compute model values
            lyapunov_exponent[process_times_index] = np.trapz(log_rho_array/t * log_rho_pdf, x=log_rho_array)
            lyapunov_exponent_2[process_times_index] = np.nanmean(compute_lyapunov_exponent(eigvals[:, 0], eigvals[:, 1], eigvals[:, 2], t))
            log_rho_var[process_times_index] = np.trapz((log_rho_array - lyapunov_exponent[process_times_index] * t)**2 * log_rho_pdf, x=log_rho_array)
            # done
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=process_times[-1]), flush=True)
        np.savetxt("{}__sc_{}__time_c_var.csv".format(object_name, str(sc).replace(".", "o")), np.column_stack((process_times, c_var)), delimiter=",", header="time, c_var")
        np.savetxt("{}__time_lyapunov_exponent.csv".format(object_name), np.column_stack((process_times, lyapunov_exponent)), delimiter=",", header="time, lyapunov_exponent")
        np.savetxt("{}__time_lyapunov_exponent_2.csv".format(object_name), np.column_stack((process_times, lyapunov_exponent_2)), delimiter=",", header="time, lyapunov_exponent_2")
        np.savetxt("{}__time_log_rho_var.csv".format(object_name), np.column_stack((process_times, log_rho_var)), delimiter=",", header="time, log_rho_var")
        print("\tINFO: Done processing object {name}.".format(name=object_name), flush=True)
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
