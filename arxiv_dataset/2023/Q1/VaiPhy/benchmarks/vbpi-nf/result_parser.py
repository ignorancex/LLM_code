import csv
import numpy as np

orig_results_path = "results/"

out_file = open(orig_results_path + 'vbpi-nf_output.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
headers = ["data_seed", "taxa", "pos",
           "Max LL", "Mean LL", "Std LL", "Mean MargLL", "Std MargLL"]
num_columns = len(headers)
tsv_writer.writerow(headers)

num_missing = 0
num_present = 0

data_seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_taxa_list = [10, 20, 40]
n_pos_list = [300, 1000]

real_dataset_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for data_seed in data_seed_list:
    for n_taxa in n_taxa_list:
        for n_pos in n_pos_list:
            results_path = orig_results_path + "data_seed_" + str(data_seed) + "_taxa_" + str(n_taxa) + "_pos_" + str(
                n_pos) + "/"
            log_fname = results_path + "vbpi-nf_summ_out.txt"
            # print(log_fname)

            try:
                log_file = open(log_fname, 'r')
                ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_std = np.nan, np.nan, np.nan, np.nan, np.nan
                Lines = log_file.readlines()
                for line in Lines:
                    if "Lower Bound." in line:
                        temp_line = line.rsplit(" ")
                        marg_ll_mean = temp_line[4]
                        marg_ll_std = temp_line[7][:-1]

                    if "LogLikelihood." in line:
                        temp_line = line.rsplit(" ")
                        ll_mean = temp_line[3]
                        ll_std = temp_line[6][:-1]

                    if "Max Loglikelihood:" in line:
                        ll_max = line.rsplit(" ")[-1][:-1]

                tsv_writer.writerow([data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_std])

                log_file.close()
                num_present += 1
            except:
                num_missing += 1

for dataset_id in real_dataset_ids:

    results_path = orig_results_path + "DS" + str(dataset_id) + "/"
    log_fname = results_path + "vbpi-nf_summ_out.txt"
    # print(log_fname)

    data_seed = "DS" + str(dataset_id)
    n_taxa, n_pos = np.nan, np.nan

    try:
        log_file = open(log_fname, 'r')
        ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_std = np.nan, np.nan, np.nan, np.nan, np.nan
        Lines = log_file.readlines()
        for line in Lines:
            if "Lower Bound." in line:
                temp_line = line.rsplit(" ")
                marg_ll_mean = temp_line[4]
                marg_ll_std = temp_line[7][:-1]

            if "LogLikelihood." in line:
                temp_line = line.rsplit(" ")
                ll_mean = temp_line[3]
                ll_std = temp_line[6][:-1]

            if "Max Loglikelihood:" in line:
                ll_max = line.rsplit(" ")[-1][:-1]

        tsv_writer.writerow([data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_std])

        log_file.close()
        num_present += 1
    except:
        num_missing += 1

print("Number of total files: ", num_present + num_missing)
print("Number of present files: ", num_present)
print("Number of missing files: ", num_missing)

out_file.close()