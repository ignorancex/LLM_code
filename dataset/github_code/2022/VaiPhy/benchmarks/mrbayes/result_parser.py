import csv
import numpy as np

orig_results_path = "results/"

out_file = open(orig_results_path + 'mrbayes_output.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
headers = ["data_seed", "taxa", "pos",
           "Max LL", "Mean LL", "Std LL", "Mean MargLL", "Mean MargLL OurCalc", "Std MargLL OurCalc"]
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

            log_fname = results_path + "mrbayes_ss_out.txt"
            #print(log_fname)

            try:
                log_file = open(log_fname, 'r')
                ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc, marg_ll_std_ourcalc = np.NINF, np.nan, np.nan, np.nan, np.nan, np.nan

                line_counter = 0
                line_flag = False

                marg_ll_list = []

                Lines = log_file.readlines()
                for line in Lines:
                    if "Mean:" in line:
                        if "BurnIn" not in line:
                            marg_ll_mean = line.rsplit(" ")[-1][:-1]
                        else:
                            if "(0.25)" in line:
                                temp_line = line.rsplit(" ")
                                ll_mean = temp_line[5]
                                ll_std = temp_line[8][:-1]

                    if "Max LL:" in line:
                        ll = float(line.rsplit(" ")[-1][:-1])
                        if ll > ll_max:
                            ll_max = ll

                    if "Run   Marginal likelihood (ln)" in line:
                        line_counter = 0
                        line_flag = True

                    if line_flag and line_counter > 1:
                        marg_ll_list.append(float(line.rsplit(" ")[-4]))

                        if line_counter == 11:
                            line_flag = False
                        # print(line.rsplit(" "))

                    line_counter += 1

                if len(marg_ll_list) > 0:
                    marg_ll_mean_ourcalc = np.mean(marg_ll_list)
                    marg_ll_std_ourcalc = np.std(marg_ll_list)

                tsv_writer.writerow(
                    [data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc,
                     marg_ll_std_ourcalc])

                log_file.close()
                num_present += 1
            except:
                num_missing += 1

for data_seed in data_seed_list:
    for n_taxa in n_taxa_list:
        for n_pos in n_pos_list:
            results_path = orig_results_path + "data_seed_" + str(data_seed) + "_taxa_" + str(n_taxa) + "_pos_" + str(
                n_pos) + "/"

            log_fname = results_path + "mrbayes_out.txt"
            # print(log_fname)

            try:
                log_file = open(log_fname, 'r')
                ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc, marg_ll_std_ourcalc = np.NINF, np.nan, np.nan, np.nan, np.nan, np.nan

                line_counter = 0
                line_flag = False

                marg_ll_list = []

                Lines = log_file.readlines()
                for line in Lines:
                    if "Mean:" in line:
                        if "BurnIn" not in line:
                            marg_ll_mean = line.rsplit(" ")[-1][:-1]
                        else:
                            if "(0.25)" in line:
                                temp_line = line.rsplit(" ")
                                ll_mean = temp_line[5]
                                ll_std = temp_line[8][:-1]

                    if "Max LL:" in line:
                        ll = float(line.rsplit(" ")[-1][:-1])
                        if ll > ll_max:
                            ll_max = ll

                    if "Run   Marginal likelihood (ln)" in line:
                        line_counter = 0
                        line_flag = True

                    if line_flag and line_counter > 1:
                        marg_ll_list.append(float(line.rsplit(" ")[-4]))

                        if line_counter == 11:
                            line_flag = False
                        # print(line.rsplit(" "))

                    line_counter += 1

                if len(marg_ll_list) > 0:
                    marg_ll_mean_ourcalc = np.mean(marg_ll_list)
                    marg_ll_std_ourcalc = np.std(marg_ll_list)

                tsv_writer.writerow(
                    [data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc,
                     marg_ll_std_ourcalc])

                log_file.close()
                num_present += 1
            except:
                num_missing += 1

for dataset_id in real_dataset_ids:

    results_path = orig_results_path + "DS" + str(dataset_id) + "/"
    log_fname = results_path + "mrbayes_out.txt"
    # print(log_fname)

    data_seed = "DS" + str(dataset_id)
    n_taxa, n_pos = np.nan, np.nan

    try:
        log_file = open(log_fname, 'r')
        ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc, marg_ll_std_ourcalc = np.NINF, np.nan, np.nan, np.nan, np.nan, np.nan

        line_counter = 0
        line_flag = False

        marg_ll_list = []

        Lines = log_file.readlines()
        for line in Lines:
            if "Mean:" in line:
                if "BurnIn" not in line:
                    marg_ll_mean = line.rsplit(" ")[-1][:-1]
                else:
                    if "(0.25)" in line:
                        temp_line = line.rsplit(" ")
                        ll_mean = temp_line[5]
                        ll_std = temp_line[8][:-1]

            if "Max LL:" in line:
                ll = float(line.rsplit(" ")[-1][:-1])
                if ll > ll_max:
                    ll_max = ll

            if "Run   Marginal likelihood (ln)" in line:
                line_counter = 0
                line_flag = True

            if line_flag and line_counter > 1:
                marg_ll_list.append(float(line.rsplit(" ")[-4]))

                if line_counter == 11:
                    line_flag = False
                # print(line.rsplit(" "))

            line_counter += 1

        if len(marg_ll_list) > 0:
            marg_ll_mean_ourcalc = np.mean(marg_ll_list)
            marg_ll_std_ourcalc = np.std(marg_ll_list)

        tsv_writer.writerow([data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc,
                             marg_ll_std_ourcalc])

        log_file.close()
        num_present += 1
    except:
        num_missing += 1

for dataset_id in real_dataset_ids:

    results_path = orig_results_path + "DS" + str(dataset_id) + "/"
    log_fname = results_path + "mrbayes_ss_out.txt"
    #print(log_fname)

    data_seed = "DS" + str(dataset_id)
    n_taxa, n_pos = np.nan, np.nan

    try:
        log_file = open(log_fname, 'r')
        ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc, marg_ll_std_ourcalc = np.NINF, np.nan, np.nan, np.nan, np.nan, np.nan

        line_counter = 0
        line_flag = False

        marg_ll_list = []

        Lines = log_file.readlines()
        for line in Lines:
            if "Mean:" in line:
                if "BurnIn" not in line:
                    marg_ll_mean = line.rsplit(" ")[-1][:-1]
                else:
                    if "(0.25)" in line:
                        temp_line = line.rsplit(" ")
                        ll_mean = temp_line[5]
                        ll_std = temp_line[8][:-1]

            if "Max LL:" in line:
                ll = float(line.rsplit(" ")[-1][:-1])
                if ll > ll_max:
                    ll_max = ll

            if "Run   Marginal likelihood (ln)" in line:
                line_counter = 0
                line_flag = True

            if line_flag and line_counter > 1:
                marg_ll_list.append(float(line.rsplit(" ")[-4]))

                if line_counter == 11:
                    line_flag = False
                # print(line.rsplit(" "))

            line_counter += 1

        if len(marg_ll_list) > 0:
            marg_ll_mean_ourcalc = np.mean(marg_ll_list)
            marg_ll_std_ourcalc = np.std(marg_ll_list)

        tsv_writer.writerow([data_seed, n_taxa, n_pos, ll_max, ll_mean, ll_std, marg_ll_mean, marg_ll_mean_ourcalc, marg_ll_std_ourcalc])

        log_file.close()
        num_present += 1
    except:
        num_missing += 1

print("Number of total files: ", num_present + num_missing)
print("Number of present files: ", num_present)
print("Number of missing files: ", num_missing)

out_file.close()