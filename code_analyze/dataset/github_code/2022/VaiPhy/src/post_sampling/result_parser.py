import csv
import numpy as np

orig_results_path = "../results/"

out_file = open(orig_results_path + 'csmc_synthetic_output.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
headers = ["data_seed", "taxa", "pos", "n_particles", "vaiphy_seed",
           "csmc_n_rep", "csmc_dist_rate", "csmc_n_particles", "tree_merger",
           "branch_sampler", "mean_logZ", "std_logZ", "mean_LL", "std_LL", "max_LL",
           "Consensus_LL_05", "Consensus_LL_01"]
num_columns = len(headers)
tsv_writer.writerow(headers)

num_missing = 0
num_present = 0

data_seed_list = [1, 2, 3]
vaiphy_seed_list = [13]
n_taxa_list = [10, 20, 40]
n_particles_list = [32, 128]
n_pos_list = [300, 1000]
csmc_n_rep_list = [10]
n_csmc_particle_list = [2048]
csmc_poi_rate_list = [0, 4]
csmc_merger_list = ["uniform", "naive_bootstrap_0", "naive_bootstrap_2"]
csmc_branch_list = ["naive", "prior", "naive_w_labels", "same_parent"]

for data_seed in data_seed_list:
    for n_taxa in n_taxa_list:
        for n_pos in n_pos_list:
            for init in ["nj_phyml"]:
                for samp in ["slantis"]:
                    for br in ["ml"]:
                        for ng in ["0.8"]:
                            for n_particle in n_particles_list:
                                for vaiphy_seed in vaiphy_seed_list:
                                    for csmc_n_rep in csmc_n_rep_list:
                                        for csmc_poi_rate in csmc_poi_rate_list:
                                            for n_csmc_particle in n_csmc_particle_list:
                                                for csmc_merger in csmc_merger_list:
                                                    for csmc_branch in csmc_branch_list:
                                                        results_path = orig_results_path \
                                                                       + "data_seed_" + str(data_seed) \
                                                                       + "_taxa_" + str(n_taxa) \
                                                                       + "_pos_" + str(n_pos) + "/"
                                                        results_path += "model_vaiphy_init_" + init + "_samp_" + samp \
                                                                        + "_branch_" + br + "_ng_" + ng + "/"
                                                        results_path += "S_" + str(n_particle) \
                                                                        + "_seed_" + str(vaiphy_seed) + "/"

                                                        log_fname = results_path + "csmc_vaiphy_S_" + str(n_particle) \
                                                                        + "_seed_" + str(vaiphy_seed) \
                                                                        + "_K_" + str(n_csmc_particle) \
                                                                        + "_poisson_" + str(csmc_poi_rate) \
                                                                        + "_rep_" + str(csmc_n_rep) \
                                                                        + "_" + csmc_branch + "_" + csmc_merger + ".log"
                                                        #print(log_fname)

                                                        try:
                                                            log_file = open(log_fname, 'r')
                                                        except:
                                                            num_missing += 1
                                                            continue

                                                        mean_logZ, std_logZ, mean_ll, std_ll, max_ll \
                                                            = np.nan, np.nan, np.nan, np.nan, np.nan
                                                        consensus_05_ll, consensus_01_ll = np.nan, np.nan

                                                        num_warn = 0
                                                        Lines = log_file.readlines()
                                                        for line in Lines:
                                                            #print(line)
                                                            #print(line.rsplit(" "))

                                                            if "Mean marginal loglikelihood estimate" in line:
                                                                mean_logZ = float(line.rsplit(" ")[-1][:-1])

                                                            if "Std marginal loglikelihood estimate" in line:
                                                                std_logZ = float(line.rsplit(" ")[-1][:-1])

                                                            if "Particle LLs" in line:
                                                                mean_ll = float(line.rsplit(" ")[9][:-1])
                                                                std_ll = float(line.rsplit(" ")[11][:-1])
                                                                max_ll = float(line.rsplit(" ")[-1][:-1])

                                                            if "Consensus tree's Log-Likelihood (cutoff=0.5)" in line:
                                                                consensus_05_ll = float(line.rsplit(" ")[-1][:-1])

                                                            if "Consensus tree's Log-Likelihood (cutoff=0.1)" in line:
                                                                consensus_01_ll = float(line.rsplit(" ")[-1][:-1])

                                                        log_file.close()
                                                        num_present += 1

                                                        tsv_writer.writerow([data_seed, n_taxa, n_pos,
                                                                             n_particle, vaiphy_seed,
                                                                             csmc_n_rep, csmc_poi_rate, n_csmc_particle,
                                                                             csmc_merger, csmc_branch,
                                                                             mean_logZ, std_logZ,
                                                                             mean_ll, std_ll, max_ll,
                                                                             consensus_05_ll, consensus_01_ll])
                                                        log_file.close()

print("Number of total files: ", num_present + num_missing)
print("Number of present files: ", num_present)
print("Number of missing files: ", num_missing)

out_file.close()
