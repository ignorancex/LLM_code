#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# scipy
import scipy as sp
import scipy.stats
# custom lib
import libpost

BINS_NB = 64
BINS_U_RANGE = (-0.21*4, 0.21*4)
BINS_J_RANGE = (-11.5, 11.5)
BINS_W_RANGE = (0, 11.5)
BINS_RATIO_RANGE = (0, 5.0)

def parse():
    parser = argparse.ArgumentParser(description='Tracers post processing')
    parser.add_argument('names', nargs="*", help='specify the object names')
    return parser.parse_args()

def main(names):
    if not names:
        print("INFO: Reading objects...", flush=True)
        names = libpost.get_object_names()
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    time = time[:-1]
    print("INFO: Done.", flush=True)
    for name in names:
        print("INFO: Computing {} flow sampled statistics...".format(name), flush=True)
        rms_flow_velocity = np.empty(time.size)
        rms_vorticity = np.empty(time.size)
        average_flow_velocity = np.empty((time.size, 3))
        average_flow_velocity_intensity = np.empty(time.size)
        average_flow_velocity_abs = np.empty((time.size, 3))
        average_streching = np.empty(time.size)
        average_sym_grads_norm = np.empty(time.size)
        average_grads_norm = np.empty(time.size)
        average_vorticity = np.empty((time.size, 3))
        average_vorticity_intensity = np.empty(time.size)
        average_skew_grads_norm = np.empty(time.size)
        average_norm_ratio = np.empty(time.size)
        flow_velocity_pdf = np.empty((time.size, BINS_NB, 3))
        flow_vorticity_pdf = np.empty((time.size, BINS_NB, 3))
        flow_vorticity_intensity_pdf = np.empty((time.size, BINS_NB, 1))
        flow_gradients_pdf = np.empty((time.size, BINS_NB, 9))
        norm_ratio_pdf = np.empty((time.size, BINS_NB, 1))
        for index in range(time.size):
            t = time[index]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read flow
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract velocity
            flow_velocity_0 = libpost.get_object_properties(obj, [".*__u_0"])
            flow_velocity_1 = libpost.get_object_properties(obj, [".*__u_1"])
            flow_velocity_2 = libpost.get_object_properties(obj, [".*__u_2"])
            flow_velocity = np.empty((flow_velocity_0["value"].size, 3))
            flow_velocity[:, 0] = flow_velocity_0["value"]
            flow_velocity[:, 1] = flow_velocity_1["value"]
            flow_velocity[:, 2] = flow_velocity_2["value"]
            del flow_velocity_0;
            del flow_velocity_1;
            del flow_velocity_2;
            # extract gradients
            flow_gradients_0_0 = libpost.get_object_properties(obj, [".*__j_0_0"])
            flow_gradients_0_1 = libpost.get_object_properties(obj, [".*__j_0_1"])
            flow_gradients_0_2 = libpost.get_object_properties(obj, [".*__j_0_2"])
            flow_gradients_1_0 = libpost.get_object_properties(obj, [".*__j_1_0"])
            flow_gradients_1_1 = libpost.get_object_properties(obj, [".*__j_1_1"])
            flow_gradients_1_2 = libpost.get_object_properties(obj, [".*__j_1_2"])
            flow_gradients_2_0 = libpost.get_object_properties(obj, [".*__j_2_0"])
            flow_gradients_2_1 = libpost.get_object_properties(obj, [".*__j_2_1"])
            flow_gradients_2_2 = libpost.get_object_properties(obj, [".*__j_2_2"])
            flow_gradients = np.empty((flow_gradients_0_0["value"].size, 3, 3))
            flow_gradients[:, 0, 0] = flow_gradients_0_0["value"]
            flow_gradients[:, 0, 1] = flow_gradients_0_1["value"]
            flow_gradients[:, 0, 2] = flow_gradients_0_2["value"]
            flow_gradients[:, 1, 0] = flow_gradients_1_0["value"]
            flow_gradients[:, 1, 1] = flow_gradients_1_1["value"]
            flow_gradients[:, 1, 2] = flow_gradients_1_2["value"]
            flow_gradients[:, 2, 0] = flow_gradients_2_0["value"]
            flow_gradients[:, 2, 1] = flow_gradients_2_1["value"]
            flow_gradients[:, 2, 2] = flow_gradients_2_2["value"]
            del flow_gradients_0_0;
            del flow_gradients_0_1;
            del flow_gradients_0_2;
            del flow_gradients_1_0;
            del flow_gradients_1_1;
            del flow_gradients_1_2;
            del flow_gradients_2_0;
            del flow_gradients_2_1;
            del flow_gradients_2_2;
            # more gradients
            sym_flow_gradients = 0.5 * (flow_gradients + flow_gradients.transpose((0, 2, 1)))
            skew_flow_gradients = 0.5 * (flow_gradients - flow_gradients.transpose((0, 2, 1)))
            # vorticity
            vorticity = np.empty((skew_flow_gradients.shape[0], 3))
            vorticity[:, 0] = 2.0 * -skew_flow_gradients[:, 1, 2]
            vorticity[:, 1] = 2.0 * skew_flow_gradients[:, 0, 2]
            vorticity[:, 2] = 2.0 * -skew_flow_gradients[:, 0, 1]
            # compute statistics
            average_flow_velocity[index, :] = np.average(flow_velocity, axis=0)
            average_flow_velocity_abs[index, :] = np.average(np.abs(flow_velocity), axis=0)
            rms_flow_velocity[index] = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.sum(np.square(flow_velocity - average_flow_velocity[index, :]), axis=1)))
            average_vorticity[index, :] = np.average(vorticity, axis=0)
            rms_vorticity[index] = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.sum(np.square(vorticity - average_vorticity[index, :]), axis=1)))
            average_flow_velocity_intensity[index] = np.average(np.linalg.norm(flow_velocity, axis=1))
            average_streching[index] = np.average(np.linalg.norm(sym_flow_gradients, axis=(1, 2)))
            average_sym_grads_norm[index] = np.average(np.linalg.norm(sym_flow_gradients, axis=(1, 2)))
            average_skew_grads_norm[index] = np.average(np.linalg.norm(skew_flow_gradients, axis=(1, 2)))
            average_grads_norm[index] = np.average(np.linalg.norm(flow_gradients, axis=(1, 2)))
            average_norm_ratio[index] = np.nanmean(np.linalg.norm(skew_flow_gradients, axis=(1, 2))/np.linalg.norm(sym_flow_gradients, axis=(1, 2)))
            average_vorticity_intensity[index] = np.average(np.linalg.norm(vorticity, axis=1))
            # pdfs
            flow_velocity_pdf[index, :, 0], bins_u = np.histogram(flow_velocity[:, 0], bins=BINS_NB, range=BINS_U_RANGE, density=True)
            flow_velocity_pdf[index, :, 1], bins_u = np.histogram(flow_velocity[:, 1], bins=BINS_NB, range=BINS_U_RANGE, density=True)
            flow_velocity_pdf[index, :, 2], bins_u = np.histogram(flow_velocity[:, 2], bins=BINS_NB, range=BINS_U_RANGE, density=True)
            
            flow_vorticity_pdf[index, :, 0], bins_j = np.histogram(vorticity[:, 0], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_vorticity_pdf[index, :, 1], bins_j = np.histogram(vorticity[:, 1], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_vorticity_pdf[index, :, 2], bins_j = np.histogram(vorticity[:, 2], bins=BINS_NB, range=BINS_J_RANGE, density=True)

            flow_vorticity_intensity_pdf[index, :, 0], bins_w = np.histogram(np.linalg.norm(vorticity, axis=1), bins=BINS_NB, range=BINS_W_RANGE, density=True)

            norm_ratio_pdf[index, :, 0], bins_ratio = np.histogram(np.linalg.norm(skew_flow_gradients, axis=(1, 2))/np.linalg.norm(sym_flow_gradients, axis=(1, 2)), bins=BINS_NB, range=BINS_RATIO_RANGE, density=True)
            
            flow_gradients_pdf[index, :, 0], bins_j = np.histogram(flow_gradients[:, 0, 0], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 1], bins_j = np.histogram(flow_gradients[:, 0, 1], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 2], bins_j = np.histogram(flow_gradients[:, 0, 2], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 3], bins_j = np.histogram(flow_gradients[:, 1, 0], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 4], bins_j = np.histogram(flow_gradients[:, 1, 1], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 5], bins_j = np.histogram(flow_gradients[:, 1, 2], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 6], bins_j = np.histogram(flow_gradients[:, 2, 0], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 7], bins_j = np.histogram(flow_gradients[:, 2, 1], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            flow_gradients_pdf[index, :, 8], bins_j = np.histogram(flow_gradients[:, 2, 2], bins=BINS_NB, range=BINS_J_RANGE, density=True)
            # print
            print("\tINFO: Done.", flush=True)
        rms_flow_velocity = np.average(rms_flow_velocity)
        rms_vorticity = np.average(rms_vorticity)
        average_flow_velocity = np.average(average_flow_velocity, axis=0)
        average_flow_velocity_abs = np.average(average_flow_velocity_abs, axis=0)
        average_flow_velocity_intensity = np.average(average_flow_velocity_intensity)
        average_streching = np.average(average_streching)
        average_sym_grads_norm = np.average(average_sym_grads_norm)
        average_skew_grads_norm = np.average(average_skew_grads_norm)
        average_grads_norm = np.average(average_grads_norm)
        average_vorticity = np.average(average_vorticity, axis=0)
        average_vorticity_intensity = np.average(average_vorticity)
        flow_velocity_pdf = np.average(flow_velocity_pdf, axis=0)
        flow_gradients_pdf = np.average(flow_gradients_pdf, axis=0)
        flow_vorticity_pdf = np.average(flow_vorticity_pdf, axis=0)
        flow_vorticity_intensity_pdf = np.average(flow_vorticity_intensity_pdf, axis=0)
        norm_ratio_pdf = np.average(norm_ratio_pdf, axis=0)
        average_norm_ratio = np.average(average_norm_ratio)
        print("INFO: Done...", flush=True)
        print("INFO: Saving...", flush=True)
        np.savetxt("{name}__flow_sampled_averages.csv".format(name=name), np.column_stack([rms_flow_velocity, rms_vorticity, average_flow_velocity_intensity, average_streching, average_vorticity_intensity, average_flow_velocity[0], average_flow_velocity[1], average_flow_velocity[2], average_vorticity[0], average_vorticity[1], average_vorticity[2], average_flow_velocity_abs[0], average_flow_velocity_abs[1], average_flow_velocity_abs[2], average_grads_norm, average_sym_grads_norm, average_skew_grads_norm, average_norm_ratio]), delimiter=",", header="rms_flow_velocity,rms_vorticity,average_flow_velocity_intensity,average_streching,average_vorticity_intensity,u_x,u_y,u_z,w_x,w_y,w_z,a_u_x,a_u_y,a_u_z,grads_norm,sym_grads_norm,skew_grads_norm,skew_grads_norm/sym_grads_norm")
        np.savetxt("{name}__flow_velocity_sampled_pdfs.csv".format(name=name), np.column_stack([0.5 * (bins_u[:-1] + bins_u[1:]), flow_velocity_pdf]), delimiter=",", header="value,p_u_x,p_u_y,p_u_z")
        np.savetxt("{name}__flow_gradients_sampled_pdfs.csv".format(name=name), np.column_stack([0.5 * (bins_j[:-1] + bins_j[1:]), flow_gradients_pdf]), delimiter=",", header="value,p_j_0_0,p_j_0_1,p_j_0_2,p_j_1_0,p_j_1_1,p_j_1_2,p_j_2_0,p_j_2_1,p_j_2_2")
        np.savetxt("{name}__flow_vorticity_sampled_pdfs.csv".format(name=name), np.column_stack([0.5 * (bins_j[:-1] + bins_j[1:]), flow_vorticity_pdf]), delimiter=",", header="value,p_w_0,p_w_1,p_w_2")
        np.savetxt("{name}__flow_vorticity_intensity_sampled_pdfs.csv".format(name=name), np.column_stack([0.5 * (bins_w[:-1] + bins_w[1:]), flow_vorticity_intensity_pdf]), delimiter=",", header="value,p_w")
        np.savetxt("{name}__flow_norm_ratio_pdfs.csv".format(name=name), np.column_stack([0.5 * (bins_ratio[:-1] + bins_ratio[1:]), norm_ratio_pdf]), delimiter=",", header="value,p_norm_ratio")
        print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.names)
