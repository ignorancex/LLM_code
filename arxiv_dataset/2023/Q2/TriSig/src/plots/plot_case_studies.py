import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator
import random
import os.path as path
from ast import literal_eval

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from TriSig import *
from process_datasets.glycine import *
from process_datasets.mouse_genes import *
from process_datasets.batches import *

curr_dir = os.getcwd()

I_D_D = True

def scatter_plot_triclusters(triclusters, test, save_folder,nl, X_axis_label, Y_axis_label, Z_axis_label):
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
    #crit_val = 0.05

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


def read_results_delta_and_trias(file):
    fp = open(file, 'r')
    content = fp.readlines()
    triclusters = []
    for line in content:
        aux = line.split("{'rows': ")[1]
        aux = aux.split(", 'columns': ")
        rows = literal_eval(aux[0])

        aux = aux[1].split(", 'times': ")
        columns = literal_eval(aux[0])

        aux = aux[1].split("}")
        times = literal_eval(aux[0])

        triclusters.append({
            "rows": rows,
            "columns": columns,
            "times": times
        })

    return triclusters


def read_results_zaki(file):
    fp = open(file, 'r')
    content = fp.readlines()

    content = list(map(lambda x: x.strip(), content))

    flag = False
    triclusters = list()
    for c in content:
        if c.startswith("====================================================================================================================="):
            flag = True
        elif c.startswith("|T|x|S|x|G|") and flag:
            l = c.split(":")[1].split('x')
            tri = {
                "rows": [],
                "columns": [],
                "times": []
            }
        elif c.startswith("Time") and flag:
            time = int(c.split(":")[1].strip())
            if time not in tri["times"]:
                tri["times"].append(time)
        elif c.startswith("S-") and flag:
            smp = list(map(lambda x: int(x.strip()), c.split("S-")[1:]))
            for s in smp:
                if s not in tri["columns"]:
                    tri["columns"].append(s)
        elif c.startswith("G") and flag:
            pc = c.split('\t')[0]
            pci = int(pc.split("-")[1])
            if pci not in tri["rows"]:
                tri["rows"].append(pci)
        elif c.startswith("Cluster") and flag:
            triclusters.append(tri)
    return triclusters


results_folder = path.abspath(path.join(__file__, "../../.."))+"\\results\\"

result_files_zaki = [
    results_folder+"zaki\\triclusters_batches.txt",
    results_folder+"zaki\\triclusters_glycine.txt",
    results_folder+"zaki\\triclusters_mouse_genes.txt"
]

result_files_trias = [
    results_folder+"trias\\triclusters_batches.txt",
    results_folder+"trias\\triclusters_glycine.txt",
    results_folder+"trias\\triclusters_mouse_genes.txt"
]

result_files_delta = [
    results_folder+"delta_trimax\\triclusters_batches.txt",
    results_folder+"delta_trimax\\triclusters_glycine.txt",
    results_folder+"delta_trimax\\triclusters_mouse_genes.txt"
]

curr_dir = os.getcwd()

######################################################### Case studies scatter plots #########################################################

# Batches
print("Processing Batch dataset results")
batch_data = read_batchs(discretized_batch_files)

tri_results_zaki_batch = read_results_zaki(result_files_zaki[0])
for i in range(len(tri_results_zaki_batch)):
    temp = tri_results_zaki_batch[i]["rows"]
    tri_results_zaki_batch[i]["rows"] = tri_results_zaki_batch[i]["columns"]
    tri_results_zaki_batch[i]["columns"] = temp
tri_results_delta_batch = read_results_delta_and_trias(result_files_delta[0])
tri_results_trias_batch = read_results_delta_and_trias(result_files_trias[0])
for i in range(len(tri_results_trias_batch)):
    temp = tri_results_trias_batch[i]["rows"]
    tri_results_trias_batch[i]["rows"] = tri_results_trias_batch[i]["columns"]
    tri_results_trias_batch[i]["columns"] = temp

test_class = TriSig(batch_data)


scatter_plot_triclusters(tri_results_zaki_batch, test_class.y_ind_z_markov, curr_dir + "\\zaki_batch_ind_time.png", 7, "Batchs", "Variables", "Time points")
scatter_plot_triclusters(tri_results_trias_batch, test_class.y_ind_z_dep, curr_dir + "\\trias_batch_ind_dep.png", 7, "Batchs", "Variables", "Time points")
scatter_plot_triclusters(tri_results_delta_batch, test_class.y_ind_z_dep, curr_dir + "\\delta_batch_ind_dep.png", 7, "Batchs", "Variables", "Time points")

scatter_plot_triclusters(tri_results_zaki_batch, test_class.y_dep_z_markov, curr_dir + "\\zaki_batch_dep_time.png", 7, "Batchs", "Variables", "Time points")
scatter_plot_triclusters(tri_results_trias_batch, test_class.y_dep_z_dep, curr_dir + "\\trias_batch_dep_dep.png", 7, "Batchs", "Variables", "Time points")
scatter_plot_triclusters(tri_results_delta_batch, test_class.y_dep_z_dep, curr_dir + "\\delta_batch_dep_dep.png", 7, "Batchs", "Variables", "Time points")
print("Finished plotting scatter plots for Batch dataset")


# Glycine
print("Processing Glycine dataset results")
glycine_data = read_glycine(glycine_discretized_files, False)

tri_results_zaki_glycine = read_results_zaki(result_files_zaki[1])
for i in range(len(tri_results_zaki_glycine)):
    temp = tri_results_zaki_glycine[i]["rows"]
    tri_results_zaki_glycine[i]["rows"] = tri_results_zaki_glycine[i]["columns"]
    tri_results_zaki_glycine[i]["columns"] = temp

tri_results_trias_glycine = read_results_delta_and_trias(result_files_trias[1])

tri_results_delta_glycine = read_results_delta_and_trias(result_files_delta[1])

test_class = TriSig(glycine_data)

#Scatter plots
scatter_plot_triclusters(tri_results_zaki_glycine, test_class.y_ind_z_markov, curr_dir + "\\zaki_glycine_ind_time.png", 5, "Subjects", "Units", "Time Points")
scatter_plot_triclusters(tri_results_trias_glycine, test_class.y_ind_z_dep, curr_dir + "\\trias_glycine_ind_dep.png", 5, "Subjects", "Units", "Time Points")
scatter_plot_triclusters(tri_results_delta_glycine, test_class.y_ind_z_dep, curr_dir + "\\delta_glycine_ind_dep.png", 5, "Subjects", "Units", "Time Points")

scatter_plot_triclusters(tri_results_zaki_glycine, test_class.y_dep_z_markov, curr_dir + "\\zaki_glycine_dep_time.png", 5, "Subjects", "Units", "Time Points")
scatter_plot_triclusters(tri_results_trias_glycine, test_class.y_dep_z_dep, curr_dir + "\\trias_glycine_dep_dep.png", 5, "Subjects", "Units", "Time Points")
scatter_plot_triclusters(tri_results_delta_glycine, test_class.y_dep_z_dep, curr_dir + "\\delta_glycine_dep_dep.png", 5, "Subjects", "Units", "Time Points")

print("Finished scatterplot glycine")

# Mouse
print("Processing mouse")
mouse_data = read_mouse_data(discretized_mouse_files, False)

tri_results_zaki_mouse = read_results_zaki(result_files_zaki[2])
for i in range(len(tri_results_zaki_mouse)):
    temp = tri_results_zaki_mouse[i]["rows"]
    tri_results_zaki_mouse[i]["rows"] = tri_results_zaki_mouse[i]["columns"]
    tri_results_zaki_mouse[i]["columns"] = temp

tri_results_delta_mouse = read_results_delta_and_trias(result_files_delta[2])

tri_results_trias_mouse = read_results_delta_and_trias(result_files_trias[2])

test_class = TriSig(mouse_data)

scatter_plot_triclusters(tri_results_zaki_mouse, test_class.y_ind_z_markov, curr_dir + "\\zaki_mouse_ind_time.png", 3, "Subjects", "Genes", "Time Points")
scatter_plot_triclusters(tri_results_trias_mouse, test_class.y_ind_z_dep, curr_dir + "\\trias_mouse_ind_dep.png", 3, "Subjects", "Genes", "Time Points")
scatter_plot_triclusters(tri_results_delta_mouse, test_class.y_ind_z_dep, curr_dir + "\\delta_mouse_ind_dep.png", 3, "Subjects", "Genes", "Time Points")

scatter_plot_triclusters(tri_results_zaki_mouse, test_class.y_dep_z_markov, curr_dir + "\\zaki_mouse_dep_time.png", 3, "Subjects", "Genes", "Time Points")
scatter_plot_triclusters(tri_results_trias_mouse, test_class.y_dep_z_dep, curr_dir + "\\trias_mouse_dep_dep.png", 3, "Subjects", "Genes", "Time Points")
scatter_plot_triclusters(tri_results_delta_mouse, test_class.y_dep_z_dep, curr_dir + "\\delta_mouse_dep_dep.png", 3, "Subjects", "Genes", "Time Points")

print("Finished scatterplot mouse")




