import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.font_manager
import pandas as pd
import numpy as np
from TriSig import *
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MaxNLocator
import os.path as path


curr_dir = os.getcwd()

I_D_D = False


def scatter_plot_triclusters(triclusters, dataset, test_i, save_folder,nl, X_axis_label, Y_axis_label, Z_axis_label):
    test = None
    if test_i == 0:
        test = TriSig(dataset).y_ind_z_dep
    elif test_i == 1:
        test = TriSig(dataset).y_dep_z_dep
    elif test_i == 2:
        test = TriSig(dataset).y_ind_z_markov
    elif test_i == 3:
        test = TriSig(dataset).y_dep_z_markov
    elif test_i == 4:
        test = TriSig(dataset).y_ind_z_dep
    elif test_i == 5:
        test = TriSig(dataset).y_dep_z_dep
    elif test_i == 6:
        test = TriSig(dataset).y_ind_z_markov
    elif test_i == 7:
        test = TriSig(dataset).y_dep_z_markov
    rows = []
    columns = []
    times = []
    pvalues = []
    for t in triclusters:
        p_value = test(t, I_D_D)
        t["pvalue"] = p_value
        pvalues.append(p_value)
        rows.append(len(t["rows"]))
        columns.append(len(t["columns"]))
        times.append(len(t["times"]))

    crit_val = TriSig.hochberg_critical_value(pvalues)
    if crit_val == 0.0:
        crit_val = sys.float_info.min

    non_sig = [t for t in triclusters if t["pvalue"] > crit_val]
    sig = [t for t in triclusters if t["pvalue"] <= crit_val]

    non_sig_rows_avg = round(float(np.mean([len(t["rows"]) for t in non_sig])), 2) if len(non_sig) != 0 else 0.0
    non_sig_cols_avg = round(float(np.mean([len(t["columns"]) for t in non_sig])), 2) if len(non_sig) != 0 else 0.0
    non_sig_time_avg = round(float(np.mean([len(t["times"]) for t in non_sig])), 2) if len(non_sig) != 0 else 0.0

    sig_rows_avg = round(float(np.mean([len(t["rows"]) for t in sig])), 2) if len(sig) != 0 else 0.0
    sig_columns_avg = round(float(np.mean([len(t["columns"]) for t in sig])), 2) if len(sig) != 0 else 0.0
    sig_times_avg = round(float(np.mean([len(t["times"]) for t in sig])), 2) if len(sig) != 0 else 0.0

    txt = "Sig. Triclusters: |" + u'I\u0304'+ "| = "+str(sig_rows_avg)+" , |" + u'J\u0304'+ "| = "+str(sig_columns_avg)+" , |" + u'K\u0304'+ "| = "+str(sig_times_avg)+" , Nr = "+str(len(sig))+\
        "\nNon Sig. Triclusters: |" + u'I\u0304'+ "| = "+str(non_sig_rows_avg)+" , |" + u'J\u0304'+ "| = "+str(non_sig_cols_avg)+" , |" + u'K\u0304'+ "| = "+str(non_sig_time_avg)+" , Nr = "+str(len(non_sig))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    wiggle = np.array(random.sample(range(len(triclusters)*2), len(triclusters)))
    wiggle = ((wiggle-np.min(wiggle))/(np.max(wiggle)-np.min(wiggle)))/10

    for i in range(len(rows)):
        color = "blue"
        if pvalues[i] < 0.05:
            color = "orange"
        if pvalues[i] <= crit_val:
            color = "red"
        ax.scatter(
            times[i] + (wiggle[i] * (1 if random.randint(0, 1) == 0 else -1)),
            columns[i] + (wiggle[i] * (1 if random.randint(0, 1) == 0 else -1)),
            rows[i] + (wiggle[i] * (1 if random.randint(0, 1) == 0 else -1)),
            marker="o",
            c=color,
            alpha=0.70,
            s=50
        )

    ax.set_zlabel(X_axis_label, fontsize="x-large", labelpad=20, rotation="vertical")
    ax.set_xlabel(Z_axis_label, fontsize="x-large", labelpad=20)
    ax.set_ylabel(Y_axis_label, fontsize="x-large", labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1, int(np.max(times))+1)
    ax.set_ylim(1, int(np.max(columns))+1)
    ax.set_zlim(1, int(np.max(rows))+1)
    fig.text(0, -0.1, txt, ha='left', fontsize="x-large", transform=ax.transAxes)
    plt.title("p-value ≲ "+str('{:.3g}'.format(crit_val))+"\n|L| = "+str(nl), fontsize="xx-large")
    plt.savefig(save_folder, dpi=300, transparent=True)
    plt.close('all')


def read_tri_tab_files(file, y_dimension, z_dimension):
    dataframes = []
    f = open(file)
    lines = f.readlines()
    lines = lines[1:]
    temp_dataframes = [[] for i in range(z_dimension)]
    for line in lines:
        values = np.array(line.replace("\n","").split("\t")[1:])
        values = np.split(values, z_dimension)
        for i in range(len(values)):
            temp_dataframes[i].append(values[i])

    for a_i in range(len(temp_dataframes)):
        dataframes.append(pd.DataFrame(np.row_stack(temp_dataframes[a_i])).astype(float))

    return dataframes


def read_synthetic_tric_planted(file):
    extracted_triclusters = []
    f_t = open(file)
    lines = f_t.readlines()
    for line in lines:
        line = line.split(" ")
        if "(" in line[0]:
            x = None
            y = None
            z = None
            for val in line:
                if "X" in val:
                    aux = val.split("=")[1].replace("[", "").replace("]", "").split(",")[:-1]
                    x = aux
                if "Y" in val:
                    aux = val.split("=")[1].replace("[", "").replace("]", "").split(",")[:-1]
                    y = aux
                if "Z" in val:
                    aux = val.split("=")[1].replace("[", "").replace("]", "").split(",")[:-1]
                    z = aux
            extracted_triclusters.append({
                "rows": list(map(int, x)),
                "columns": list(map(int, y)),
                "times": list(map(int, z))
            })
    return extracted_triclusters


data_folder = path.abspath(path.join(__file__, "../../.."))+"\\data\\synthetic_data\\"

syntetic_data_1000_3 = [
    data_folder+"uniform\\non_contiguity\\3\\1000_data.tsv",
    data_folder+"uniform\\non_contiguity\\3\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\3\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\3\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\3\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\3\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\3\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\3\\1000_data.tsv",
]

syntetic_tri_1000_3 = [
    data_folder+"uniform\\non_contiguity\\3\\1000_trics.txt",
    data_folder+"uniform\\non_contiguity\\3\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\3\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\3\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\3\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\3\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\3\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\3\\1000_trics.txt",
]

syntetic_tri_1000_names_3 = [
    "uniform_1000_ind_dep_3.png",
    "uniform_1000_dep_dep_3.png",
    "uniform_1000_ind_time_3.png",
    "uniform_1000_dep_time_3.png",
    "gaussian_1000_ind_dep_3.png",
    "gaussian_1000_dep_dep_3.png",
    "gaussian_1000_ind_time_3.png",
    "gaussian_1000_dep_time_3.png"
]

syntetic_data_1000_5 = [
    data_folder+"uniform\\non_contiguity\\5\\1000_data.tsv",
    data_folder+"uniform\\non_contiguity\\5\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\5\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\5\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\5\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\5\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\5\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\5\\1000_data.tsv",
]

syntetic_tri_1000_5 = [
    data_folder+"uniform\\non_contiguity\\5\\1000_trics.txt",
    data_folder+"uniform\\non_contiguity\\5\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\5\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\5\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\5\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\5\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\5\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\5\\1000_trics.txt",
]

syntetic_tri_1000_names_5 = [
    "uniform_1000_ind_dep_5.png",
    "uniform_1000_dep_dep_5.png",
    "uniform_1000_ind_time_5.png",
    "uniform_1000_dep_time_5.png",
    "gaussian_1000_ind_dep_5.png",
    "gaussian_1000_dep_dep_5.png",
    "gaussian_1000_ind_time_5.png",
    "gaussian_1000_dep_time_5.png"
]

syntetic_data_1000_7 = [
    data_folder+"uniform\\non_contiguity\\7\\1000_data.tsv",
    data_folder+"uniform\\non_contiguity\\7\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\7\\1000_data.tsv",
    data_folder+"uniform\\contiguity\\7\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\7\\1000_data.tsv",
    data_folder+"gaussian\\non_contiguity\\7\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\7\\1000_data.tsv",
    data_folder+"gaussian\\contiguity\\7\\1000_data.tsv",
]

syntetic_tri_1000_7 = [
    data_folder+"uniform\\non_contiguity\\7\\1000_trics.txt",
    data_folder+"uniform\\non_contiguity\\7\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\7\\1000_trics.txt",
    data_folder+"uniform\\contiguity\\7\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\7\\1000_trics.txt",
    data_folder+"gaussian\\non_contiguity\\7\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\7\\1000_trics.txt",
    data_folder+"gaussian\\contiguity\\7\\1000_trics.txt",
]

syntetic_tri_1000_names_7 = [
    "uniform_1000_ind_dep_7.png",
    "uniform_1000_dep_dep_7.png",
    "uniform_1000_ind_time_7.png",
    "uniform_1000_dep_time_7.png",
    "gaussian_1000_ind_dep_7.png",
    "gaussian_1000_dep_dep_7.png",
    "gaussian_1000_ind_time_7.png",
    "gaussian_1000_dep_time_7.png"
]

for i in range(len(syntetic_data_1000_3)):

    print("iteration "+str(i))

    df_1000_3 = read_tri_tab_files(syntetic_data_1000_3[i],50,50)
    print("Finished reading Dataset 1000_3")
    tri_1000_3 = read_synthetic_tric_planted(syntetic_tri_1000_3[i])
    print("Finished reading planted triclusters 1000_3")
    scatter_plot_triclusters(tri_1000_3,df_1000_3,i,curr_dir+"\\"+syntetic_tri_1000_names_3[i],3, "X", "Y", "Z")
    print("Finished scatterplot 1000_3")


    df_1000_5 = read_tri_tab_files(syntetic_data_1000_5[i],50,50)
    print("Finished reading Dataset 1000_5")
    tri_1000_5 = read_synthetic_tric_planted(syntetic_tri_1000_5[i])
    print("Finished reading planted triclusters 1000_5")
    scatter_plot_triclusters(tri_1000_5,df_1000_5,i,curr_dir+"\\"+syntetic_tri_1000_names_5[i],5, "X", "Y", "Z")
    print("Finished scatterplot 1000_5")


    df_1000_7 = read_tri_tab_files(syntetic_data_1000_7[i],50,50)
    print("Finished reading Dataset 1000_7")
    tri_1000_7 = read_synthetic_tric_planted(syntetic_tri_1000_7[i])
    print("Finished reading planted triclusters 1000_7")
    scatter_plot_triclusters(tri_1000_7,df_1000_7,i,curr_dir+"\\"+syntetic_tri_1000_names_7[i],7, "X", "Y", "Z")
    print("Finished scatterplot 1000_7")






