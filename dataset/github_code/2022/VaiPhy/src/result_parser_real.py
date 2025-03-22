import csv
import numpy as np

orig_results_path = "../results/"

out_file = open(orig_results_path + 'vaiphy_real_output.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
headers = ["data_seed", "taxa", "pos", "True LL",
           "max_iter", "tree-init", "tree-propagation", "branch-length",
           "NG", "n_particles", "vaiphy_seed", "iwelbo n_runs", "iwelbo n_samples",
           "best_iter", "best_LB", "NJ_LL", "NJ_RF",
           "Edmonds_LL", "Edmonds_RF", "Edmonds_max_RF", "Edmonds_LL_bifur", "Edmonds_RF_bifur", "Edmonds_max_RF_bifur",
           "Selected_LL", "Selected_RF", "Selected_max_RF", "Selected_LL_bifur", "Selected_RF_bifur",
           "Consensus_LL_05", "Consensus_RF_05", "Consensus_max_RF_05",
           "Consensus_LL_01", "Consensus_RF_01", "Consensus_max_RF_01"]
num_columns = len(headers)
tsv_writer.writerow(headers)

num_missing = 0
num_present = 0

exp_id_list = [1, 2, 3, 4, 5]
vaiphy_seed_list = [13, 42]
n_particles_list = [32, 64, 128]
n_iwelbo_runs = 5
n_iwelbo_samples = 25

for exp_id in exp_id_list:
    for init in ["nj_phyml"]:
        for samp in ["slantis"]:
            for br in ["ml"]:
                for ng in ["0.8"]:
                    for n_particle in n_particles_list:
                        for vaiphy_seed in vaiphy_seed_list:
                            results_path = orig_results_path + "DS" + str(exp_id) + "/"
                            results_path += "model_vaiphy_init_" + init + "_samp_" + samp \
                                            + "_branch_" + br + "_ng_" + ng + "/"
                            results_path += "S_" + str(n_particle) + "_seed_" + str(vaiphy_seed) + "/"

                            log_fname = results_path + "vaiphy_S_" + str(n_particle) \
                                        + "_seed_" + str(vaiphy_seed) + ".log"
                            #print(log_fname)

                            try:
                                log_file = open(log_fname, 'r')
                            except:
                                num_missing += 1
                                continue

                            data_seed = "DS" + str(exp_id)
                            n_taxa, n_pos = np.nan, np.nan
                            true_ll, best_iter, best_lb, nj_ll, nj_rf, \
                            edmonds_ll, edmonds_rf, edmonds_max_rf, edmonds_ll_bifur, edmonds_rf_bifur, edmonds_max_rf_bifur, \
                            selected_ll, selected_rf, selected_max_rf, selected_ll_bifur, selected_rf_bifur, \
                            consensus_05_ll, consensus_05_rf, consensus_05_max_rf, \
                            consensus_01_ll, consensus_01_rf, consensus_01_max_rf \
                                = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                                  np.nan, np.nan, np.nan, np.nan
                            best_iter = -1

                            num_warn = 0
                            Lines = log_file.readlines()
                            for line in Lines:
                                #print(line)
                                #print(line.rsplit(" "))

                                if "True Tree Log-Likelihood" in line:
                                    true_ll = float(line.rsplit(" ")[-1][:-1])

                                if "Best lower bound:" in line:
                                    best_iter = int(float(line.rsplit(" ")[-2]))
                                    best_lb = float(line.rsplit(" ")[-5][:-1])

                                if "NJ Tree. RF" in line:
                                    nj_rf = int(float(line.rsplit(" ")[8]))

                                if "NJ-tree after optimization LL" in line:
                                    nj_ll = float(line.rsplit(" ")[-1][:-1])

                                if "Edmonds tree. RF" in line:
                                    edmonds_rf = int(float(line.rsplit(" ")[8]))
                                    edmonds_max_rf = int(float(line.rsplit(" ")[11]))

                                if "Edmonds tree (bifurcated). RF" in line:
                                    edmonds_rf_bifur = int(float(line.rsplit(" ")[9]))
                                    edmonds_max_rf_bifur = int(float(line.rsplit(" ")[12]))

                                if "Edmonds tree's Log-Likelihood" in line:
                                    edmonds_ll = float(line.rsplit(" ")[-1][:-1])

                                if "Edmonds tree's Log-Likelihood (bifurcated)" in line:
                                    edmonds_ll_bifur = float(line.rsplit(" ")[-1][:-1])

                                if "Selected tree. RF" in line:
                                    selected_rf = int(float(line.rsplit(" ")[8]))
                                    selected_max_rf = int(float(line.rsplit(" ")[11]))

                                if "Selected tree (bifurcated)" in line:
                                    selected_rf_bifur = int(float(line.rsplit(" ")[9]))

                                if "Selected tree log-likelihood" in line:
                                    selected_ll = float(line.rsplit(" ")[-1][:-1])

                                if "Selected tree log-likelihood" in line:
                                    selected_ll_bifur = float(line.rsplit(" ")[-1][:-1])

                                if "Consensus tree (cutoff=0.5). RF" in line:
                                    consensus_05_rf = int(float(line.rsplit(" ")[9]))
                                    consensus_05_max_rf = int(float(line.rsplit(" ")[12]))

                                if "Consensus tree (cutoff=0.1). RF" in line:
                                    consensus_01_rf = int(float(line.rsplit(" ")[9]))
                                    consensus_01_max_rf = int(float(line.rsplit(" ")[12]))

                                if "Consensus tree's Log-Likelihood (cutoff=0.5)" in line:
                                    consensus_05_ll = float(line.rsplit(" ")[-1][:-1])

                                if "Consensus tree's Log-Likelihood (cutoff=0.1)" in line:
                                    consensus_01_ll = float(line.rsplit(" ")[-1][:-1])

                                if "max_iter" in line:
                                    max_iter = int(float(line.rsplit(" ")[15][:-1]))

                            log_file.close()
                            num_present += 1

                            tsv_writer.writerow([data_seed, n_taxa, n_pos, true_ll,
                                                 max_iter, init, samp, br, ng, n_particle, vaiphy_seed,
                                                 n_iwelbo_runs, n_iwelbo_samples,
                                                 best_iter, best_lb, nj_ll, nj_rf,
                                                 edmonds_ll, edmonds_rf, edmonds_max_rf,
                                                 edmonds_ll_bifur, edmonds_rf_bifur, edmonds_max_rf_bifur,
                                                 selected_ll, selected_rf, selected_max_rf,
                                                 selected_ll_bifur, selected_rf_bifur,
                                                 consensus_05_ll, consensus_05_rf, consensus_05_max_rf,
                                                 consensus_01_ll, consensus_01_rf, consensus_01_max_rf])
                            log_file.close()

print("Number of total files: ", num_present + num_missing)
print("Number of present files: ", num_present)
print("Number of missing files: ", num_missing)

out_file.close()
