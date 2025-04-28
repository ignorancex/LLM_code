import os
import time
import pickle
import argparse
from pmle import pMLEGMM
from CTDGMR.utils import *
from CTDGMR.distance import GMM_CTD


def GlobalLocalEstimation(inputs):
    """

    the function used to conduct simulations

    inputs: a list of length 6, containing configurations to conduct simulations
    """
    # n_split = 100
    # n_split = 10

    random_state = inputs[0]
    total_sample_size = inputs[1]

    # true popualtion parameter values
    true_means = inputs[2]
    true_covs = inputs[3]
    true_weights = inputs[4]

    K, D, _ = true_covs.shape
    true_precisions = np.empty((K, D, D))
    for k, cov in enumerate(true_covs):
        true_precisions[k, :, :] = np.linalg.inv(cov)

    # output directory
    save_folder = inputs[5]
    n_split = inputs[6]
    # ------------------------------
    # Sample from true mixture
    # ------------------------------
    GMM_sample, _ = rmixGaussian(true_means, true_covs, true_weights,
                                 total_sample_size, random_state)
    true_predicted_labels = label_predict(true_weights, true_means, true_covs,
                                          GMM_sample)
    # # ------------------------------
    # # Global pMLE
    # # ------------------------------
    # gmm = pMLEGMM(
    #     n_components=K,
    #     cov_reg=1.0 / np.sqrt(total_sample_size),
    #     covariance_type="full",
    #     max_iter=10000,
    #     n_init=1,
    #     tol=1e-6,
    #     weights_init=true_weights,
    #     means_init=true_means,
    #     precisions_init=true_precisions,
    #     random_state=0,
    #     verbose=0,
    #     verbose_interval=1,
    # )
    # start_time = time.time()
    # gmm.fit(GMM_sample)
    # global_time = time.time() - start_time
    # pmle_means, pmle_covs, pmle_weights = gmm.means_, gmm.covariances_, gmm.weights_
    # global2true_W1 = GMM_CTD(
    #     [pmle_means, true_means],
    #     [pmle_covs, true_covs],
    #     [pmle_weights, true_weights],
    #     "W1",
    # )
    # global_predicted_label = label_predict(pmle_weights, pmle_means, pmle_covs,
    #                                        GMM_sample)
    # global_ARI = ARI(true_predicted_labels, global_predicted_label)
    # global_ll = gmm.score(GMM_sample) * GMM_sample.shape[0]

    # ------------------------------
    # split and combine
    # ------------------------------
    local_means = [np.empty((K, D)) for _ in range(n_split)]
    local_covs = [np.empty((K, D, D)) for _ in range(n_split)]
    local_weights = [np.empty((K, )) for _ in range(n_split)]

    locals2true_W1 = [None] * n_split
    local_converg = [None] * n_split
    local_ARI = [None] * n_split
    local_ll = [None] * n_split
    local_time = [None] * n_split
    local_iter = [None] * n_split
    local_converg = [None] * n_split

    np.random.seed(random_state)
    index = np.random.permutation(total_sample_size)
    GMM_sample = GMM_sample[index]
    true_predicted_labels = true_predicted_labels[index]
    local_length = total_sample_size // n_split

    # -------------------------------------------------------------------
    # split sample to different machines and fit mixture on each machine
    # -------------------------------------------------------------------
    for split in range(n_split):
        local = GMM_sample[(split * local_length):((split + 1) * local_length)]
        # random warm start first and then start from the true initial value
        # gmmk = pMLEGMM(
        #     n_components=K,
        #     cov_reg=1.0 / np.sqrt(local.shape[0]),
        #     covariance_type="full",
        #     max_iter=50,
        #     n_init=10,
        #     tol=1e-6,
        #     #    weights_init=true_weights,
        #     #    means_init=true_means,
        #     #    precisions_init=true_precisions,
        #     random_state=1,
        #     verbose=0,
        #     verbose_interval=1,
        #     warm_start=True)
        # start_time = time.time()
        # gmmk.fit(local)
        # gmmk.max_iter = 10000
        # gmmk.fit(local)

        # also initialize with true parameter value
        gmmk = pMLEGMM(n_components=K,
                       cov_reg=1.0 / np.sqrt(local.shape[0]),
                       covariance_type="full",
                       max_iter=10000,
                       n_init=1,
                       tol=1e-6,
                       weights_init=true_weights,
                       means_init=true_means,
                       precisions_init=true_precisions,
                       random_state=0,
                       verbose=0,
                       verbose_interval=1)
        start_time = time.time()
        gmmk.fit(local)

        # if gmmk_true.lower_bound_ > gmmk.lower_bound_:
        #     gmmk = gmmk_true
        local_timek = time.time() - start_time
        local_iterk = gmmk.n_iter_
        local_convergk = gmmk.converged_
        local_time[split] = local_timek
        local_iter[split] = local_iterk
        local_converg[split] = local_convergk
        local_pmle_means, local_pmle_covs, local_pmle_weights = (
            gmmk.means_, gmmk.covariances_, gmmk.weights_)
        local_means[split] = local_pmle_means
        local_covs[split] = local_pmle_covs
        local_weights[split] = local_pmle_weights

        locals2true_W1[split] = GMM_CTD(
            [local_pmle_means, true_means],
            [local_pmle_covs, true_covs],
            [local_pmle_weights, true_weights],
            "W1",
        )
        local_converg[split] = local_convergk
        local_resp, local_predicted_label = label_predict(
            local_pmle_weights,
            local_pmle_means,
            local_pmle_covs,
            GMM_sample,
            return_resp=True,
        )
        local_ARI[split] = ARI(true_predicted_labels, local_predicted_label)
        local_ll[split] = np.log(local_resp.sum(1)).sum(0)

    # save output data

    output_data = {
        # "globaltime": global_time,
        "local_time": local_time,
        # "global2true_W1": global2true_W1,
        "local2true_W1": locals2true_W1,
        # "global_ARI": global_ARI,
        "local_ARI": local_ARI,
        # "global": (pmle_means, pmle_covs, pmle_weights),
        "local": (local_means, local_covs, local_weights),
        # "global_ll": global_ll,
        "local_ll": local_ll,
    }

    save_file = os.path.join(
        save_folder,
        "case_" + str(random_state) + "_nsplit_" + str(n_split) + "_ncomp_" +
        str(K) + "_d_" + str(D) + "_ss_" + str(sample_size) + ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output_data, f)
    f.close()


def main(seed, sample_size, overlap, n_split):
    # fix the number of components to be 5
    num_components = 5
    dimension = 10

    base_dir = "./generated_pop/true_param"
    true_weights = np.loadtxt(
        os.path.join(
            base_dir,
            "weights_seed_" + str(seed) + "_ncomp_" + str(num_components) +
            "_d_" + str(dimension) + "_maxoverlap_" + str(overlap) + ".txt",
        ))

    true_means = np.loadtxt(
        os.path.join(
            base_dir,
            "means_seed_" + str(seed) + "_ncomp_" + str(num_components) +
            "_d_" + str(dimension) + "_maxoverlap_" + str(overlap) + ".txt",
        )).reshape((-1, dimension))

    true_covs = np.loadtxt(
        os.path.join(
            base_dir,
            "covs_seed_" + str(seed) + "_ncomp_" + str(num_components) +
            "_d_" + str(dimension) + "_maxoverlap_" + str(overlap) + ".txt",
        ))
    true_covs = true_covs.T.reshape((-1, dimension, dimension))

    params = [
        seed, sample_size, true_means, true_covs, true_weights, save_folder,
        n_split
    ]
    GlobalLocalEstimation(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset split GMM estimator comparison")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--ss",
                        type=int,
                        default=200000,
                        help="Total sample size from a GMM")
    parser.add_argument("--n_split",
                        type=int,
                        default=100,
                        help="# of local machines")
    parser.add_argument("--overlap",
                        type=float,
                        default=0.01,
                        help="degree of overlap")
    args = parser.parse_args()
    sample_size = int(args.ss)
    seed = args.seed
    overlap = args.overlap
    n_split = args.n_split

    save_folder = os.path.join("./output/save_data/", "ss_" + str(sample_size),
                               "overlap_" + str(overlap))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    main(seed, sample_size, overlap, n_split)
