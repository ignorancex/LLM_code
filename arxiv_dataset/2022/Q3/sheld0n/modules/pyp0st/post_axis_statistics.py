#!/usr/bin/env python3

# command line program
import argparse
import copy
import numpy as np
import scipy as sp
import scipy.optimize
# internal modules
import libpost

DIRECTION = np.array([[1.0, 0.0, 0.0]]).T
TAU_KOLMOGOROV = 0.12842
TAU = 1.0 * TAU_KOLMOGOROV
SYM_TAU = TAU
ASYM_TAU = TAU

BIN_RANGE = (0.0, np.pi)
BIN_NB = 'auto'

STEP_T = 1

def parse():
    parser = argparse.ArgumentParser(description='Computes particles axis statistics.')
    return parser.parse_args()

def main():
    print("INFO: Post processing orientation statistics of lagrangian objects.", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time = libpost.get_time();
    print("INFO: Done.", flush=True)
    print("INFO: Reading objects properties...", flush=True)
    # get gradients
    objects_j_0_0 = libpost.get_objects_properties(["j_0_0"], object_names)
    objects_j_0_1 = libpost.get_objects_properties(["j_0_1"], object_names)
    objects_j_0_2 = libpost.get_objects_properties(["j_0_2"], object_names)
    objects_j_1_0 = libpost.get_objects_properties(["j_1_0"], object_names)
    objects_j_1_1 = libpost.get_objects_properties(["j_1_1"], object_names)
    objects_j_1_2 = libpost.get_objects_properties(["j_1_2"], object_names)
    objects_j_2_0 = libpost.get_objects_properties(["j_2_0"], object_names)
    objects_j_2_1 = libpost.get_objects_properties(["j_2_1"], object_names)
    objects_j_2_2 = libpost.get_objects_properties(["j_2_2"], object_names)
    # get axis
    objects_axis_0 = libpost.get_objects_properties(["axis_0"], object_names)
    objects_axis_1 = libpost.get_objects_properties(["axis_1"], object_names)
    objects_axis_2 = libpost.get_objects_properties(["axis_2"], object_names)
    print("INFO: Done.", flush=True)
    print("INFO: Reducing time data...", flush=True)
    for object_name in objects_j_0_0:
        # get gradients
        objects_j_0_0[object_name]["value"] = objects_j_0_0[object_name]["value"][::STEP_T, :]
        objects_j_0_1[object_name]["value"] = objects_j_0_1[object_name]["value"][::STEP_T, :]
        objects_j_0_2[object_name]["value"] = objects_j_0_2[object_name]["value"][::STEP_T, :]
        objects_j_1_0[object_name]["value"] = objects_j_1_0[object_name]["value"][::STEP_T, :]
        objects_j_1_1[object_name]["value"] = objects_j_1_1[object_name]["value"][::STEP_T, :]
        objects_j_1_2[object_name]["value"] = objects_j_1_2[object_name]["value"][::STEP_T, :]
        objects_j_2_0[object_name]["value"] = objects_j_2_0[object_name]["value"][::STEP_T, :]
        objects_j_2_1[object_name]["value"] = objects_j_2_1[object_name]["value"][::STEP_T, :]
        objects_j_2_2[object_name]["value"] = objects_j_2_2[object_name]["value"][::STEP_T, :]
        # axis data
        objects_axis_0[object_name]["value"] = objects_axis_0[object_name]["value"][::STEP_T, :]
        objects_axis_1[object_name]["value"] = objects_axis_1[object_name]["value"][::STEP_T, :]
        objects_axis_2[object_name]["value"] = objects_axis_2[object_name]["value"][::STEP_T, :]
    print("INFO: Done.", flush=True)
    # reduce axis to a scalar
    objects_axis_angle_surf = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    objects_axis_angle_surf_sym = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    objects_axis_angle_surf_asym = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    objects_axis_angle_surf_orth_z = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    objects_axis_angle_surf_sym_orth_z = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    objects_axis_angle_surf_asym_orth_z = {object_name:{"value":np.empty(objects_axis_0[object_name]["value"].shape), "info":objects_axis_0[object_name]["info"].copy()} for object_name in objects_axis_0}
    print("INFO: Computing orientation statistics...", flush=True)
    for object_name in objects_axis_angle_surf:
        # value
        print("INFO: Processing {}".format(object_name), flush=True)
        for i in range(0, objects_axis_angle_surf[object_name]["value"].shape[0]):
            print("INFO: Processing {i}/{nb}".format(i=i, nb=objects_axis_angle_surf[object_name]["value"].shape[0]), flush=True)
            for j in range(0, objects_axis_angle_surf[object_name]["value"].shape[1]):
                # set gradients
                gradients = np.empty((3, 3))
                gradients[0, 0] = objects_j_0_0[object_name]["value"][i, j]
                gradients[0, 1] = objects_j_0_1[object_name]["value"][i, j]
                gradients[0, 2] = objects_j_0_2[object_name]["value"][i, j]
                gradients[1, 0] = objects_j_1_0[object_name]["value"][i, j]
                gradients[1, 1] = objects_j_1_1[object_name]["value"][i, j]
                gradients[1, 2] = objects_j_1_2[object_name]["value"][i, j]
                gradients[2, 0] = objects_j_2_0[object_name]["value"][i, j]
                gradients[2, 1] = objects_j_2_1[object_name]["value"][i, j]
                gradients[2, 2] = objects_j_2_2[object_name]["value"][i, j]
                sym_gradients = 0.5 * (gradients + gradients.transpose())
                asym_gradients = 0.5 * (gradients - gradients.transpose())
                # set axis
                axis = np.empty((3,1))
                axis[0] = objects_axis_0[object_name]["value"][i, j]
                axis[1] = objects_axis_1[object_name]["value"][i, j]
                axis[2] = objects_axis_2[object_name]["value"][i, j]
                axis_orth_z = axis - np.dot(axis.T, DIRECTION) * DIRECTION
                axis_orth_z = axis_orth_z / np.linalg.norm(axis_orth_z)
                # compute
                ## surf
                surf = np.dot(sp.linalg.expm(gradients * TAU).T, DIRECTION)
                surf = surf / np.linalg.norm(surf)
                objects_axis_angle_surf[object_name]["value"][i, j] = np.arccos(np.dot(surf.T, axis))
                surf_orth_z = surf - np.dot(surf.T, DIRECTION) * DIRECTION
                surf_orth_z = surf_orth_z / np.linalg.norm(surf_orth_z)
                objects_axis_angle_surf_orth_z[object_name]["value"][i, j] = np.arccos(np.dot(surf_orth_z.T, axis_orth_z))
                ## surf sym
                surf = np.dot(sp.linalg.expm(sym_gradients * SYM_TAU).T, DIRECTION)
                surf = surf / np.linalg.norm(surf)
                objects_axis_angle_surf_sym[object_name]["value"][i, j] = np.arccos(np.dot(surf.T, axis))
                surf_orth_z = surf - np.dot(surf.T, DIRECTION) * DIRECTION
                surf_orth_z = surf_orth_z / np.linalg.norm(surf_orth_z)
                objects_axis_angle_surf_sym_orth_z[object_name]["value"][i, j] = np.arccos(np.dot(surf_orth_z.T, axis_orth_z))
                ## surf asym
                surf = np.dot(sp.linalg.expm(asym_gradients * ASYM_TAU).T, DIRECTION)
                surf = surf / np.linalg.norm(surf)
                objects_axis_angle_surf_asym[object_name]["value"][i, j] = np.arccos(np.dot(surf.T, axis))
                surf_orth_z = surf - np.dot(surf.T, DIRECTION) * DIRECTION
                surf_orth_z = surf_orth_z / np.linalg.norm(surf_orth_z)
                objects_axis_angle_surf_asym_orth_z[object_name]["value"][i, j] = np.arccos(np.dot(surf_orth_z.T, axis_orth_z))
        # info
        objects_axis_angle_surf[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf") for info in objects_axis_angle_surf[object_name]["info"]]
        objects_axis_angle_surf_sym[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf_sym") for info in objects_axis_angle_surf_sym[object_name]["info"]]
        objects_axis_angle_surf_asym[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf_asym") for info in objects_axis_angle_surf_asym[object_name]["info"]]
        objects_axis_angle_surf_orth_z[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf_orth_z") for info in objects_axis_angle_surf_orth_z[object_name]["info"]]
        objects_axis_angle_surf_sym_orth_z[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf_sym_orth_z") for info in objects_axis_angle_surf_sym_orth_z[object_name]["info"]]
        objects_axis_angle_surf_asym_orth_z[object_name]["info"] = [info.replace("axis_0", "axis_angle_surf_asym_orth_z") for info in objects_axis_angle_surf_asym_orth_z[object_name]["info"]]
    print("INFO: Done.", flush=True)
    # clean
    print("INFO: Cleaning...", flush=True)
    del objects_j_0_0;
    del objects_j_0_1;
    del objects_j_0_2;
    del objects_j_1_0;
    del objects_j_1_1;
    del objects_j_1_2;
    del objects_j_2_0;
    del objects_j_2_1;
    del objects_j_2_2;
    print("INFO: Done.", flush=True)

    # compute average
    objects_average_angles = {object_name:{} for object_name in objects_axis_0}
    print("INFO: Computing average angles...", flush=True)
    for object_name in objects_average_angles:
        # value
        angles = []
        ## axis 0
        angles.append(np.average(np.arccos(objects_axis_0[object_name]["value"])))
        angles.append(np.std(np.arccos(objects_axis_0[object_name]["value"])))
        ## axis 1
        angles.append(np.average(np.arccos(objects_axis_1[object_name]["value"])))
        angles.append(np.std(np.arccos(objects_axis_1[object_name]["value"])))
        ## axis 2
        angles.append(np.average(np.arccos(objects_axis_2[object_name]["value"])))
        angles.append(np.std(np.arccos(objects_axis_2[object_name]["value"])))
        ## axis surf
        angles.append(np.average(objects_axis_angle_surf[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf[object_name]["value"]))
        ## axis surf orth z
        angles.append(np.average(objects_axis_angle_surf_orth_z[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf_orth_z[object_name]["value"]))
        ## axis surf sym
        angles.append(np.average(objects_axis_angle_surf_sym[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf_sym[object_name]["value"]))
        ## axis surf sym orth z
        angles.append(np.average(objects_axis_angle_surf_sym_orth_z[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf_sym_orth_z[object_name]["value"]))
        # axis surf asym
        angles.append(np.average(objects_axis_angle_surf_asym[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf_asym[object_name]["value"]))
        # axis surf asym orth z
        angles.append(np.average(objects_axis_angle_surf_asym_orth_z[object_name]["value"]))
        angles.append(np.std(objects_axis_angle_surf_asym_orth_z[object_name]["value"]))
        ## data
        objects_average_angles[object_name]["value"] = np.column_stack(angles)
        objects_average_angles[object_name]["info"] = [
            "average_axis_angle_0", 
            "std_axis_angle_0",
            "average_axis_angle_1", 
            "std_axis_angle_1",
            "average_axis_angle_2", 
            "std_axis_angle_2",
            "average_axis_angle_surf", 
            "std_axis_angle_surf",
            "average_axis_angle_surf_orth_z", 
            "std_axis_angle_surf_orth_z",
            "average_axis_angle_surf_sym", 
            "std_axis_angle_surf_sym",
            "average_axis_angle_surf_sym_orth_z", 
            "std_axis_angle_surf_sym_orth_z",
            "average_axis_angle_surf_asym", 
            "std_axis_angle_surf_asym",
            "average_axis_angle_surf_asym_orth_z", 
            "std_axis_angle_surf_asym_orth_z",
        ]
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving...", flush=True)
    libpost.save(objects_average_angles, "average_axis_angles")
    print("INFO: Done.", flush=True)

    # compute pdf
    objects_pdf_angle_axis_0 = {object_name:{} for object_name in objects_axis_0}
    objects_pdf_angle_axis_1 = {object_name:{} for object_name in objects_axis_1}
    objects_pdf_angle_axis_2 = {object_name:{} for object_name in objects_axis_2}
    objects_pdf_axis_angle_surf = {object_name:{} for object_name in objects_axis_angle_surf}
    objects_pdf_axis_angle_surf_sym = {object_name:{} for object_name in objects_axis_angle_surf_sym}
    objects_pdf_axis_angle_surf_asym = {object_name:{} for object_name in objects_axis_angle_surf_asym}
    objects_pdf_axis_angle_surf_orth_z = {object_name:{} for object_name in objects_axis_angle_surf_orth_z}
    objects_pdf_axis_angle_surf_sym_orth_z = {object_name:{} for object_name in objects_axis_angle_surf_sym_orth_z}
    objects_pdf_axis_angle_surf_asym_orth_z = {object_name:{} for object_name in objects_axis_angle_surf_asym_orth_z}
    print("INFO: Computing pdfs...", flush=True)
    for object_name in objects_pdf_angle_axis_0:
        # value
        ## axis 0
        angles = np.arccos(objects_axis_0[object_name]["value"])
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_angle_axis_0[object_name]["value"] = np.column_stack([bins, hist])
        ## axis 1
        angles = np.arccos(objects_axis_1[object_name]["value"])
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_angle_axis_1[object_name]["value"] = np.column_stack([bins, hist])
        ## axis 2
        angles = np.arccos(objects_axis_2[object_name]["value"])
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_angle_axis_2[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf
        angles = objects_axis_angle_surf[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf orth z
        angles = objects_axis_angle_surf_orth_z[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf_orth_z[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf sym
        angles = objects_axis_angle_surf_sym[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf_sym[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf sym orth z
        angles = objects_axis_angle_surf_sym_orth_z[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf_sym_orth_z[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf asym
        angles = objects_axis_angle_surf_asym[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf_asym[object_name]["value"] = np.column_stack([bins, hist])
        ## axis angle surf asym orth z
        angles = objects_axis_angle_surf_asym_orth_z[object_name]["value"]
        hist, bin_edges = np.histogram(angles, bins=np.histogram_bin_edges(angles, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1])/2
        objects_pdf_axis_angle_surf_asym_orth_z[object_name]["value"] = np.column_stack([bins, hist])
        # info
        objects_pdf_angle_axis_0[object_name]["info"] = ["axis_angle_0", "pdf"]
        objects_pdf_angle_axis_1[object_name]["info"] = ["axis_angle_1", "pdf"]
        objects_pdf_angle_axis_2[object_name]["info"] = ["axis_angle_2", "pdf"]
        objects_pdf_axis_angle_surf[object_name]["info"] = ["axis_angle_surf", "pdf"]
        objects_pdf_axis_angle_surf_orth_z[object_name]["info"] = ["axis_angle_surf_orth_z", "pdf"]
        objects_pdf_axis_angle_surf_sym[object_name]["info"] = ["axis_angle_surf_sym", "pdf"]
        objects_pdf_axis_angle_surf_sym_orth_z[object_name]["info"] = ["axis_angle_surf_sym_orth_z", "pdf"]
        objects_pdf_axis_angle_surf_asym[object_name]["info"] = ["axis_angle_surf_asym", "pdf"]
        objects_pdf_axis_angle_surf_asym_orth_z[object_name]["info"] = ["axis_angle_surf_asym_orth_z", "pdf"]
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving...", flush=True)
    libpost.save(objects_pdf_angle_axis_0, "pdf_axis_angle_0")
    libpost.save(objects_pdf_angle_axis_1, "pdf_axis_angle_1")
    libpost.save(objects_pdf_angle_axis_2, "pdf_axis_angle_2")
    libpost.save(objects_pdf_axis_angle_surf, "pdf_axis_angle_surf")
    libpost.save(objects_pdf_axis_angle_surf_sym, "pdf_axis_angle_surf_sym")
    libpost.save(objects_pdf_axis_angle_surf_asym, "pdf_axis_angle_surf_asym")
    libpost.save(objects_pdf_axis_angle_surf_orth_z, "pdf_axis_angle_surf_orth_z")
    libpost.save(objects_pdf_axis_angle_surf_sym_orth_z, "pdf_axis_angle_surf_sym_orth_z")
    libpost.save(objects_pdf_axis_angle_surf_asym_orth_z, "pdf_axis_angle_surf_asym_orth_z")
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
