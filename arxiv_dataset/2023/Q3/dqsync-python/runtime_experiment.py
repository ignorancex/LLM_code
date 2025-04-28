"""
Runtime experiment.
"""

import pathlib

import numpy as np

import dq_sync as dqs



def experiment_runtime(dirpath):
    # Parameters
    TEST_NAME = "experiment-runtime"
    ns = (np.arange(1, 10).reshape(-1, 1) * 10**np.arange(2, 4).reshape(1, -1)).T.flatten()[:-4]
    sigma_rs = np.array([20])
    sigma_ts = np.array([0.2])
    ps = [0.05]
    qs = [1.0]
    rep_no = 50

    # Messaging parameter
    trial_no = len(ns) * len(sigma_rs) * len(ps) * len(qs) * rep_no;

    # Set up CSV file
    fields = ["n", "sigma_r", "sigma_t", "p", "q", "rep_no", \
              "dq_mean_rerr", "dq_min_rerr", "dq_max_rerr", "dq_mean_terr", "dq_min_terr", "dq_max_terr", "dq_ev_time", "dq_rounding_time", \
              "mat_mean_rerr", "mat_min_rerr", "mat_max_rerr", "mat_mean_terr", "mat_min_terr", "mat_max_terr", "mat_ev_time", "mat_rounding_time"]
    f, dw = dqs.openCSVFile(dirpath, TEST_NAME, fields = fields)

    # Experimental loop
    row = {}
    count = 0
    for n in ns:
        row['n'] = n

        for sigma_r, sigma_t in zip(sigma_rs, sigma_ts):
            row['sigma_r'] = sigma_r
            row['sigma_t'] = sigma_t

            for p in ps:
                row['p'] = p

                for q in qs:
                    row['q'] = q

                    for rep in range(rep_no):
                        row['rep_no'] = rep

                        # Increase the coutner
                        count += 1

                        # Generate ground truth
                        ground_truth = dqs.generateGroundTruth(n)

                        # Run experiment
                        rerr_dqmat, terr_dqmat, rerr_mat, terr_mat, times \
                            = dqs.experiment(ground_truth, \
                                       sigma_r=dqs.angle2radians(sigma_r), sigma_t=sigma_t, \
                                       p=p, q=q, timeit=True)

                        # Process data
                        row["dq_mean_rerr"] = np.mean(rerr_dqmat)
                        row["dq_min_rerr"] = np.min(rerr_dqmat)
                        row["dq_max_rerr"] = np.max(rerr_dqmat)
                        row["dq_mean_terr"] = np.mean(terr_dqmat)
                        row["dq_min_terr"] = np.min(terr_dqmat)
                        row["dq_max_terr"] = np.max(terr_dqmat)

                        row["mat_mean_rerr"] = np.mean(rerr_mat)
                        row["mat_min_rerr"] = np.min(rerr_mat)
                        row["mat_max_rerr"] = np.max(rerr_mat)
                        row["mat_mean_terr"] = np.mean(terr_mat)
                        row["mat_min_terr"] = np.min(terr_mat)
                        row["mat_max_terr"] = np.max(terr_mat)
                        row = row | times

                        dw.writerow(row)

                        dqs.logmsg("Trial {:d} of {:d} completed.\n\t\tmean rerr\tmean terr\trtime\n\tDQ\t{:.3}\t\t{:.3}\t\t{:.3}\n\tMAT\t{:.3}\t\t{:.3}\t\t{:.3}".format(count, trial_no, \
                                                                                                                                                                          row["dq_mean_rerr"], row["dq_mean_terr"], row["dq_ev_time"] + row["dq_rounding_time"], \
                                                                                                                                                                          row["mat_mean_rerr"], row["mat_mean_terr"], row["mat_ev_time"] + row["mat_rounding_time"]))

    f.close()


if __name__=="__main__":
    dirpath = pathlib.Path(__file__).parent.resolve()
    print("Writing directory: \t", str(dirpath))
    experiment_runtime(dirpath=str(dirpath))

