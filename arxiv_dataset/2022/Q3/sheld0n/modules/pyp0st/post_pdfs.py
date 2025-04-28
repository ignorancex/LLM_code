#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost
# test
import matplotlib.pyplot as plt

# direction = 0 index
BINS = 1000#"auto"

def parse():
    parser = argparse.ArgumentParser(description='Computes probability density functions of flow properties measured along particle trajectories')
    return parser.parse_args()

def main():
    # get flow velocity
    objects_p_0 = libpost.get_objects_properties(["pos_0"])
    objects_p_1 = libpost.get_objects_properties(["pos_1"])
    objects_p_2 = libpost.get_objects_properties(["pos_2"])
    objects_u_0 = libpost.get_objects_properties(["u_0"])
    objects_u_1 = libpost.get_objects_properties(["u_1"])
    objects_u_2 = libpost.get_objects_properties(["u_2"])
    # get gradient matrix
    time = libpost.get_time();
    #objects_j_0_0 = libpost.get_objects_properties(["j_0_0"])
    #objects_j_0_1 = libpost.get_objects_properties(["j_0_1"])
    #objects_j_0_2 = libpost.get_objects_properties(["j_0_2"])
    #objects_j_1_0 = libpost.get_objects_properties(["j_1_0"])
    #objects_j_1_1 = libpost.get_objects_properties(["j_1_1"])
    #objects_j_1_2 = libpost.get_objects_properties(["j_1_2"])
    #objects_j_2_0 = libpost.get_objects_properties(["j_2_0"])
    #objects_j_2_1 = libpost.get_objects_properties(["j_2_1"])
    #objects_j_2_2 = libpost.get_objects_properties(["j_2_2"])
    # compute stretch and vorticity
    #objects_stretch = copy.deepcopy(objects_j_0_0)
    #objects_stretch_direction = copy.deepcopy(objects_j_0_0)
    #objects_stretch_orthogonal_direction = copy.deepcopy(objects_j_0_0)
    #objects_vorticity = copy.deepcopy(objects_j_0_0)
    #objects_vorticity_direction = copy.deepcopy(objects_j_0_0)
    #objects_vorticity_orthogonal_direction = copy.deepcopy(objects_j_0_0)
    #objects_Q = copy.deepcopy(objects_j_0_0)
    #objects_R = copy.deepcopy(objects_j_0_0)
    #objects_D = copy.deepcopy(objects_j_0_0)
    #for object_name in objects_j_0_0:
    #    # value
    #    for i in range(0, objects_j_0_0[object_name]["value"].shape[0]):
    #        for j in range(0, objects_j_0_0[object_name]["value"].shape[1]):
    #            # set J
    #            J = np.empty((3, 3))
    #            J[0, 0] = objects_j_0_0[object_name]["value"][i, j]
    #            J[0, 1] = objects_j_0_1[object_name]["value"][i, j]
    #            J[0, 2] = objects_j_0_2[object_name]["value"][i, j]
    #            J[1, 0] = objects_j_1_0[object_name]["value"][i, j]
    #            J[1, 1] = objects_j_1_1[object_name]["value"][i, j]
    #            J[1, 2] = objects_j_1_2[object_name]["value"][i, j]
    #            J[2, 0] = objects_j_2_0[object_name]["value"][i, j]
    #            J[2, 1] = objects_j_2_1[object_name]["value"][i, j]
    #            J[2, 2] = objects_j_2_2[object_name]["value"][i, j]
    #            S = 0.5 * (J + J.transpose())
    #            O = 0.5 * (J - J.transpose())
    #            # compute
    #            w, v = np.linalg.eigh(S)
    #            #objects_stretch[object_name]["value"][i, j] = np.sum(np.abs(w)) / 4.0
    #            objects_stretch_direction[object_name]["value"][i, j] = w[0] * np.abs(v[0][0]) + w[1] *np.abs(v[1][0]) + w[2] * np.abs(v[2][0])
    #            objects_stretch_orthogonal_direction[object_name]["value"][i, j] = w[0] * np.linalg.norm(v[0][1:]) + w[1] * np.linalg.norm(v[1][1:]) + w[2] * np.linalg.norm(v[2][1:])
    #            o = np.array([-O[1, 2], O[0, 2], -O[0, 1]])
    #            #objects_vorticity[object_name]["value"][i, j] = np.linalg.norm(o)
    #            objects_vorticity_direction[object_name]["value"][i, j] = o[0]
    #            objects_vorticity_orthogonal_direction[object_name]["value"][i, j] = np.linalg.norm(o[1:])
    #            Q = -1.0/2.0 * np.trace(np.linalg.matrix_power(J, 2))
    #            R = -1.0/3.0 * np.trace(np.linalg.matrix_power(J, 3))
    #            #objects_Q[object_name]["value"][i, j] = Q
    #            #objects_R[object_name]["value"][i, j] = R
    #            objects_D[object_name]["value"][i, j] = 27 * R**2 + 4 * Q**3
    #    # info
    #    #objects_stretch[object_name]["info"] = [info.replace("j_0_0", "stretch") for info in objects_stretch[object_name]["info"]]
    #    #objects_vorticity[object_name]["info"] = [info.replace("j_0_0", "vorticity") for info in objects_vorticity[object_name]["info"]]
    # compute pdfs and statistics
    objects_u_0_pdf = copy.deepcopy(objects_u_0)
    objects_u_1_pdf = copy.deepcopy(objects_u_1)
    objects_u_2_pdf = copy.deepcopy(objects_u_2)
    #objects_j_0_0_pdf = copy.deepcopy(objects_j_0_0)
    #objects_j_1_1_pdf = copy.deepcopy(objects_j_1_1)
    #objects_j_2_2_pdf = copy.deepcopy(objects_j_2_2)
    #objects_stretch_pdf = copy.deepcopy(objects_stretch)
    #objects_stretch_direction_pdf = copy.deepcopy(objects_stretch_direction)
    #objects_stretch_orthogonal_direction_pdf = copy.deepcopy(objects_stretch_orthogonal_direction)
    #objects_vorticity_pdf = copy.deepcopy(objects_vorticity)
    #objects_vorticity_direction_pdf = copy.deepcopy(objects_vorticity_direction)
    #objects_vorticity_orthogonal_direction_pdf = copy.deepcopy(objects_vorticity_orthogonal_direction)
    #objects_QR_pdf = copy.deepcopy(objects_Q) TODO
    #objects_D_pdf = copy.deepcopy(objects_D)
    objects_average_u_0 = copy.deepcopy(objects_u_0)
    objects_average_u_1 = copy.deepcopy(objects_u_1)
    objects_average_u_2 = copy.deepcopy(objects_u_2)
    #objects_average_stretch = copy.deepcopy(objects_stretch)
    #objects_average_stretch_direction = copy.deepcopy(objects_stretch_direction)
    #objects_average_stretch_orthogonal_direction = copy.deepcopy(objects_stretch_orthogonal_direction)
    #objects_average_vorticity = copy.deepcopy(objects_vorticity)
    #objects_average_vorticity_direction = copy.deepcopy(objects_vorticity_direction)
    #objects_average_vorticity_orthogonal_direction = copy.deepcopy(objects_vorticity_orthogonal_direction)
    #objects_average_D = copy.deepcopy(objects_D)
    objects_average_u_0_t = copy.deepcopy(objects_u_0)
    objects_average_u_1_t = copy.deepcopy(objects_u_1)
    objects_average_u_2_t = copy.deepcopy(objects_u_2)
    objects_std_u_0_t = copy.deepcopy(objects_u_0)
    objects_std_u_1_t = copy.deepcopy(objects_u_1)
    objects_std_u_2_t = copy.deepcopy(objects_u_2)
    for object_name in objects_u_0_pdf:
        # value
        ## u_0
        hist, bin_edges = np.histogram(objects_u_0[object_name]["value"], bins=BINS, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_u_0_pdf[object_name]["value"] = np.column_stack([bins, hist])
        objects_average_u_0[object_name]["value"] = np.array([np.average(objects_u_0[object_name]["value"])])
        objects_average_u_0_t[object_name]["value"] = np.average(objects_u_0[object_name]["value"], axis=1)
        objects_std_u_0_t[object_name]["value"] = np.std(objects_u_0[object_name]["value"], axis=1)
        ## u_1
        hist, bin_edges = np.histogram(objects_u_1[object_name]["value"], bins=BINS, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_u_1_pdf[object_name]["value"] = np.column_stack([bins, hist])
        objects_average_u_1[object_name]["value"] = np.array([np.average(objects_u_1[object_name]["value"])])
        objects_average_u_1_t[object_name]["value"] = np.average(objects_u_1[object_name]["value"], axis=1)
        objects_std_u_1_t[object_name]["value"] = np.std(objects_u_1[object_name]["value"], axis=1)
        ## u_2
        hist, bin_edges = np.histogram(objects_u_2[object_name]["value"], bins=BINS, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_u_2_pdf[object_name]["value"] = np.column_stack([bins, hist])
        objects_average_u_2[object_name]["value"] = np.array([np.average(objects_u_2[object_name]["value"])])
        objects_average_u_2_t[object_name]["value"] = np.average(objects_u_2[object_name]["value"], axis=1)
        objects_std_u_2_t[object_name]["value"] = np.std(objects_u_2[object_name]["value"], axis=1)
        ### j_0_0
        #hist, bin_edges = np.histogram(objects_j_0_0[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_j_0_0_pdf[object_name]["value"] = np.column_stack([bins, hist])
        ### j_1_1
        #hist, bin_edges = np.histogram(objects_j_0_0[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_j_0_0_pdf[object_name]["value"] = np.column_stack([bins, hist])
        ### j_2_2
        #hist, bin_edges = np.histogram(objects_j_0_0[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_j_0_0_pdf[object_name]["value"] = np.column_stack([bins, hist])
        ### stretch
        #hist, bin_edges = np.histogram(objects_stretch[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_stretch_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_stretch[object_name]["value"] = np.array([np.average(objects_stretch[object_name]["value"])])
        ### stretch direction
        #hist, bin_edges = np.histogram(objects_stretch_direction[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_stretch_direction_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_stretch_direction[object_name]["value"] = np.array([np.average(objects_stretch_direction[object_name]["value"])])
        ### stretch orthogonal direction
        #hist, bin_edges = np.histogram(objects_stretch_orthogonal_direction[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_stretch_orthogonal_direction_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_stretch_orthogonal_direction[object_name]["value"] = np.array([np.average(objects_stretch_orthogonal_direction[object_name]["value"])])
        ### vorticity
        #hist, bin_edges = np.histogram(objects_vorticity[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_vorticity_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_vorticity[object_name]["value"] = np.array([np.average(objects_vorticity[object_name]["value"])])
        ### vorticity direction
        #hist, bin_edges = np.histogram(objects_vorticity_direction[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_vorticity_direction_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_vorticity_direction[object_name]["value"] = np.array([np.average(objects_vorticity_direction[object_name]["value"])])
        ### vorticity orthogonal direction
        #hist, bin_edges = np.histogram(objects_vorticity_orthogonal_direction[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_vorticity_orthogonal_direction_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_vorticity_orthogonal_direction[object_name]["value"] = np.array([np.average(objects_vorticity_orthogonal_direction[object_name]["value"])])
        ### D
        #hist, bin_edges = np.histogram(objects_D[object_name]["value"], bins=BINS, density=True)
        #bins = (bin_edges[1:] + bin_edges[:-1])/2
        #objects_D_pdf[object_name]["value"] = np.column_stack([bins, hist])
        #objects_average_D[object_name]["value"] = np.array([np.average(objects_D[object_name]["value"])])
        # info
        objects_u_0_pdf[object_name]["info"] = ["u_0", "pdf"]
        objects_u_1_pdf[object_name]["info"] = ["u_1", "pdf"]
        objects_u_2_pdf[object_name]["info"] = ["u_2", "pdf"]
        #objects_stretch_pdf[object_name]["info"] = ["stretch", "pdf"]
        #objects_stretch_direction_pdf[object_name]["info"] = ["stretch direction", "pdf"]
        #objects_stretch_orthogonal_direction_pdf[object_name]["info"] = ["stretch orthogonal direction", "pdf"]
        #objects_vorticity_pdf[object_name]["info"] = ["vorticity", "pdf"]
        #objects_vorticity_direction_pdf[object_name]["info"] = ["vorticity direction", "pdf"]
        #objects_vorticity_orthogonal_direction_pdf[object_name]["info"] = ["vorticity orthogonal direction", "pdf"]
        #objects_D_pdf[object_name]["info"] = ["D", "pdf"]
        objects_average_u_0[object_name]["info"] = ["average u_0"]
        objects_average_u_1[object_name]["info"] = ["average u_1"]
        objects_average_u_2[object_name]["info"] = ["average u_2"]
        #objects_average_stretch[object_name]["info"] = ["average stretch"]
        #objects_average_stretch_direction[object_name]["info"] = ["average stretch direction"]
        #objects_average_stretch_orthogonal_direction[object_name]["info"] = ["average stretch orthogonal direction"]
        #objects_average_vorticity[object_name]["info"] = ["average vorticity"]
        #objects_average_vorticity_direction[object_name]["info"] = ["average vorticity direction"]
        #objects_average_vorticity_orthogonal_direction[object_name]["info"] = ["average vorticity orthogonal direction"]
        #objects_average_D[object_name]["info"] = ["D"]
        objects_average_u_0_t[object_name]["info"] = ["average u_0"]
        objects_average_u_1_t[object_name]["info"] = ["average u_1"]
        objects_average_u_2_t[object_name]["info"] = ["average u_2"]
        objects_std_u_0_t[object_name]["info"] = ["std u_0"]
        objects_std_u_1_t[object_name]["info"] = ["std u_1"]
        objects_std_u_2_t[object_name]["info"] = ["std u_2"]
    # save
    libpost.save(objects_u_0_pdf, "pdf_u_0")
    libpost.save(objects_u_1_pdf, "pdf_u_1")
    libpost.save(objects_u_2_pdf, "pdf_u_2")
    #libpost.save(objects_stretch_pdf, "pdf_stretch")
    #libpost.save(objects_stretch_direction_pdf, "pdf_stretch_direction")
    #libpost.save(objects_stretch_orthogonal_direction_pdf, "pdf_stretch_orthogonal_direction")
    #libpost.save(objects_vorticity_pdf, "pdf_vorticity")
    #libpost.save(objects_vorticity_direction_pdf, "pdf_vorticity_direction")
    #libpost.save(objects_vorticity_orthogonal_direction_pdf, "pdf_vorticity_orthogonal_direction")
    #libpost.save(objects_D_pdf, "pdf_D")
    #libpost.save(objects_average_u_0, "average_u_0")
    #libpost.save(objects_average_u_1, "average_u_1")
    #libpost.save(objects_average_u_2, "average_u_2")
    #libpost.save(objects_average_stretch, "average_stretch")
    #libpost.save(objects_average_stretch_direction, "average_stretch_direction")
    #libpost.save(objects_average_stretch_orthogonal_direction, "average_stretch_orthogonal_direction")
    #libpost.save(objects_average_vorticity, "average_vorticity")
    #libpost.save(objects_average_vorticity_direction, "average_vorticity_direction")
    #libpost.save(objects_average_vorticity_orthogonal_direction, "average_vorticity_orthogonal_direction")
    #libpost.save(objects_average_D, "average_D")
    libpost.savet(time, objects_average_u_0_t, "average_u_0")
    libpost.savet(time, objects_average_u_1_t, "average_u_1")
    libpost.savet(time, objects_average_u_2_t, "average_u_2")
    libpost.savet(time, objects_std_u_0_t, "std_u_0")
    libpost.savet(time, objects_std_u_1_t, "std_u_1")
    libpost.savet(time, objects_std_u_2_t, "std_u_2")

if __name__ == '__main__':
    args = parse()
    main()
