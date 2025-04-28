import os
import copy
import time
import pickle
import argparse
from CTDGMR.utils import *
from CTDGMR.minCTD import *
from CTDGMR.distance import GMM_CTD, GMM_L2


def robustmedian(
    subset_means,
    subset_covs,
    subset_weights,
    ground_distance="L2",
    coverage_ratio=0.5,
):
    n_split = len(subset_means)
    pairwisedist = np.zeros((n_split, n_split))
    for i in range(n_split):
        for j in range(n_split):
            if "CTD" in ground_distance:
                pairwisedist[i, j] = GMM_CTD(
                    [subset_means[i], subset_means[j]],
                    [subset_covs[i], subset_covs[j]],
                    [subset_weights[i], subset_weights[j]],
                    ground_distance=ground_distance.split("-")[1],
                    matrix=False,
                )
            else:
                pairwisedist[i, j] = np.sqrt(
                    GMM_L2(
                        [subset_means[i], subset_means[j]],
                        [subset_covs[i], subset_covs[j]],
                        [subset_weights[i], subset_weights[j]],
                    ))
    which_GMM = np.argmin(np.quantile(pairwisedist, q=coverage_ratio, axis=1))
    output = [which_GMM, pairwisedist]

    return output


def dfmr_threshold(distance_to_center, pairwisedist, which_GMM, threshold):
    half_of_num_machine = int(distance_to_center.shape[0] // 2)
    indices = np.argsort(distance_to_center)
    sorted_distance = distance_to_center[indices]
    c_sd = np.sort(pairwisedist[which_GMM])[half_of_num_machine]
    if np.sum(sorted_distance > threshold * c_sd) == 0:
        truncation = distance_to_center.shape[0]
    else:
        truncation = np.where(sorted_distance > threshold * c_sd)[0][0]

    return indices[:truncation]


def generate_attack(
    means,
    covs,
    weights,
    failure_rate,
    attack_mode,
    NumComp,
    dimension,
    NumMachine,
    seed,
    failure_type="component",
):
    contaminated_label = [np.zeros(NumComp) for _ in range(NumMachine)]
    attacked_means = copy.deepcopy(means)
    attacked_covs = copy.deepcopy(covs)
    attacked_weights = copy.deepcopy(weights)

    if failure_type == "component":
        if failure_rate != 0:
            per_comp_failure_num = int(np.floor(failure_rate * NumMachine))
            rng = np.random.default_rng(seed)
            failure_indices = []
            for k in range(NumComp):
                # randomly select the Byzantine failure components
                # The last machine is always Byzantine failure free
                failure_index = rng.choice(NumMachine - 1,
                                           per_comp_failure_num,
                                           replace=False)
                failure_indices.append(failure_index)
                # generate attacks
                for idx in failure_index:
                    if attack_mode == 1:
                        attacked_means[idx][k] = rng.normal(0,
                                                            100,
                                                            size=dimension)
                    elif attack_mode == 2:
                        attacked_covs[idx][k] = generate_random_covariance(
                            attacked_covs[idx][k], dimension, idx + k)
                    contaminated_label[idx][k] = 1
    elif failure_type == "machine":
        if failure_rate != 0:
            byzantine_machine_number = int(np.floor(failure_rate * NumMachine))
            rng = np.random.default_rng(seed)
            # randomly select the Byzantine failure machines or components
            byzantine_machine_index = rng.choice(NumMachine,
                                                 byzantine_machine_number,
                                                 replace=False)

            # generate attacks
            for byzantine_machine in byzantine_machine_index:
                if attack_mode == 1:
                    attacked_means[byzantine_machine] = rng.normal(
                        0, 100, size=(NumComp, dimension))
                elif attack_mode == 2:
                    for machine in range(NumComp):
                        attacked_covs[byzantine_machine][machine] = (
                            generate_random_covariance(
                                attacked_covs[byzantine_machine][machine],
                                dimension,
                                byzantine_machine + machine,
                            ))
                elif attack_mode == 3:
                    attacked_weights[byzantine_machine] = rng.dirichlet(
                        rng.choice(np.arange(10, 20), NumComp))
                contaminated_label[byzantine_machine] = np.ones(NumComp)

            failure_indices = []
            for k in range(NumComp):
                failure_indices.append(byzantine_machine_index)

    # concate the true contanimated label into a numpy array
    contaminated_label = np.concatenate(contaminated_label)
    return (
        attacked_means,
        attacked_covs,
        attacked_weights,
        failure_indices,
        contaminated_label,
    )


def Simulation(inputs):
    """

    the function used to conduct simulations

    inputs: a list of length 9, containing configurations to conduct simulations
    attack_mode:
    an indicator specifying the corruption mechanism to mimic the byzantine failure.
    ------ arbitary values randomly and independently generated from a (uni)multivariate normal distribution ------
        attack_mode == 1 --- sending arbitary values of mean vectors of chosen components;
        attack_mode == 2 --- sending arbitary values of covariance matrices of chosen components;
    """
    ground_distance = "KL"
    random_state = inputs[0]
    total_sample_size = inputs[1]

    # true popualtion parameter values
    true_means = inputs[2]
    true_covs = inputs[3]
    true_weights = inputs[4]
    K, D, _ = true_covs.shape

    # configurations for Byzantine failures
    save_folder = inputs[5]
    attack_mode = inputs[6]
    n_split = inputs[7]
    failure_type = inputs[8]

    # ------------------------------
    # Sample from true mixture
    # ------------------------------
    GMM_sample, _ = rmixGaussian(true_means, true_covs, true_weights,
                                 total_sample_size, random_state)
    true_predicted_labels = label_predict(true_weights, true_means, true_covs,
                                          GMM_sample)

    # ---------------------------------------------
    # Load pre-computed global and local estimators
    # ---------------------------------------------
    save_file = os.path.join(
        save_folder,
        "case_" + str(random_state) + "_nsplit_" + str(n_split) + "_ncomp_" +
        str(K) + "_d_" + str(D) + "_ss_" + str(sample_size) + ".pickle",
    )
    print(save_file)
    with open(save_file, "rb") as f:
        output_data = pickle.load(f)
    origin_local_means, origin_local_covs, origin_local_weights = output_data[
        "local"]

    # for failure_rate in [0.2]:
    for failure_rate in [0.1, 0.2, 0.3, 0.4]:
        # for failure_rate in [0.1, 0.4]:
        print("---------------", failure_rate, "--------------------")
        # -------------------------------------------------
        # Byzantine attacks with different proportions
        # -------------------------------------------------
        (
            local_means,
            local_covs,
            local_weights,
            byzantine_machine_index,
            contaminated_label,
        ) = generate_attack(
            origin_local_means,
            origin_local_covs,
            origin_local_weights,
            failure_rate,
            attack_mode,
            K,
            D,
            n_split,
            random_state,
            failure_type=failure_type,
        )
        failure_machine = np.zeros(n_split)
        for k in range(K):
            failure_machine[byzantine_machine_index[k]] = 1

        print("machine level failure {:.3f}".format(failure_machine.sum() /
                                                    n_split))
        # ------------------------------
        # COAT
        # ------------------------------
        which_GMM, pairwisedist = robustmedian(
            local_means,
            local_covs,
            local_weights,
            ground_distance="L2",
            coverage_ratio=0.5,
        )
        coat_means, coat_covs, coat_weights = (
            local_means[which_GMM],
            local_covs[which_GMM],
            local_weights[which_GMM],
        )
        coat2true_W1 = GMM_CTD(
            [coat_means, true_means],
            [coat_covs, true_covs],
            [coat_weights, true_weights],
            "W1",
        )
        coat_resp, coat_predicted_label = label_predict(
            coat_weights,
            coat_means,
            coat_covs,
            GMM_sample,
            return_resp=True,
        )
        coat_ARI = ARI(true_predicted_labels, coat_predicted_label)
        coat_ll = np.log(coat_resp.sum(1)).sum(0)

        output_data["coat_index"] = which_GMM
        output_data["coat2true_W1"] = coat2true_W1
        output_data["coat_ARI"] = coat_ARI
        output_data["coat_ll"] = coat_ll

        # ------------------------------
        # DFMR
        # ------------------------------
        distance_to_center = pairwisedist[which_GMM]

        for pow in np.linspace(0, 1 / 10, 15, endpoint=True):
            threshold = n_split**pow
            dfmr_untruncated_indices = dfmr_threshold(distance_to_center,
                                                      pairwisedist, which_GMM,
                                                      threshold)

            dfmr_gmr_optimal = np.Inf

            for i in range(n_split):
                model_temp = GMR_CTD(
                    np.concatenate([
                        local_means[index]
                        for index in dfmr_untruncated_indices
                    ]),
                    np.concatenate([
                        local_covs[index] for index in dfmr_untruncated_indices
                    ]),
                    np.concatenate([
                        local_weights[index]
                        for index in dfmr_untruncated_indices
                    ]) / dfmr_untruncated_indices.shape[0],
                    K,
                    ground_distance=ground_distance,
                    init_method="user",
                    means_init=local_means[i],
                    covs_init=local_covs[i],
                    weights_init=local_weights[i],
                )
                model_temp.iterative()
                if model_temp.obj < dfmr_gmr_optimal:
                    dfmr_gmr_optimal = model_temp.obj
                    model = model_temp
            dfmr_means, dfmr_covs, dfmr_weights = (
                model.reduced_means,
                model.reduced_covs,
                model.reduced_weights,
            )
            dfmr_2true_W1 = GMM_CTD(
                [dfmr_means, true_means],
                [dfmr_covs, true_covs],
                [dfmr_weights, true_weights],
                "W1",
            )
            dfmr_resp, dfmr_predicted_label = label_predict(
                dfmr_weights,
                dfmr_means,
                dfmr_covs,
                GMM_sample,
                return_resp=True,
            )

            dfmr_ARI = ARI(true_predicted_labels, dfmr_predicted_label)
            dfmr_ll = np.log(dfmr_resp.sum(1)).sum(0)

            output_data["dfmr_2true_W1_" + str(threshold)] = dfmr_2true_W1
            output_data["dfmr_ARI_" + str(threshold)] = dfmr_ARI
            output_data["dfmr_" + str(threshold)] = (
                dfmr_means,
                dfmr_covs,
                dfmr_weights,
            )
            output_data["dfmr_ll_" + str(threshold)] = dfmr_ll
            output_data["dfmr_trimmed_" +
                        str(threshold)] = dfmr_untruncated_indices

        # ------------------------------
        # Trimmed k-barycenter with the 50% as the trimming level
        # ------------------------------
        trim_gmr_optimal = np.Inf
        for i in range(n_split):
            model_temp = GMR_PCTD(
                np.concatenate(local_means),
                np.concatenate(local_covs),
                np.concatenate(local_weights) / n_split,
                K,
                ground_distance="KL",
                init_method="user",
                alpha=0.5,  # 50% trimming
                means_init=local_means[i],
                covs_init=local_covs[i],
                weights_init=local_weights[i],
            )
            model_temp.iterative()
            if model_temp.obj < trim_gmr_optimal:
                trim_gmr_optimal = model_temp.obj
                model = model_temp
        trim_means, trim_covs, trim_weights = (
            model.reduced_means,
            model.reduced_covs,
            model.reduced_weights,
        )
        trim2true_W1 = GMM_CTD(
            [trim_means, true_means],
            [trim_covs, true_covs],
            [trim_weights, true_weights],
            "W1",
        )
        trim_resp, trim_predicted_label = label_predict(trim_weights,
                                                        trim_means,
                                                        trim_covs,
                                                        GMM_sample,
                                                        return_resp=True)
        trim_ARI = ARI(true_predicted_labels, trim_predicted_label)
        trim_ll = np.log(trim_resp.sum(1)).sum(0)

        output_data["trim2true_W1"] = trim2true_W1
        output_data["trim_ARI"] = trim_ARI
        output_data["trim_ll"] = trim_ll
        output_data["trim_label"] = model.trimmed_label

        # ------------------------------
        # GMR without trimming
        # ------------------------------
        gmr_optimal = np.Inf
        for i in range(n_split):

            reduced_gmm_temp = GMR_CTD(
                np.concatenate(local_means),
                np.concatenate(local_covs),
                np.concatenate(local_weights) / n_split,
                K,
                ground_distance=ground_distance,
                init_method="user",
                means_init=coat_means,
                covs_init=coat_covs,
                weights_init=coat_weights,
            )
            reduced_gmm_temp.iterative()
            if reduced_gmm_temp.obj < gmr_optimal:
                gmr_optimal = reduced_gmm_temp.obj
                reduced_gmm = reduced_gmm_temp
        gmr_means, gmr_covs, gmr_weights = (
            reduced_gmm.reduced_means,
            reduced_gmm.reduced_covs,
            reduced_gmm.reduced_weights,
        )
        gmr2true_W1 = GMM_CTD(
            [gmr_means, true_means],
            [gmr_covs, true_covs],
            [gmr_weights, true_weights],
            "W1",
            False,
        )
        gmr_resp, gmr_predicted_label = label_predict(gmr_weights,
                                                      gmr_means,
                                                      gmr_covs,
                                                      GMM_sample,
                                                      return_resp=True)

        gmr_ARI = ARI(true_predicted_labels, gmr_predicted_label)
        gmr_ll = np.log(gmr_resp.sum(1)).sum(0)

        # output_data["gmr_time"] = gmr_time
        output_data["gmr2true_W1"] = gmr2true_W1
        output_data["gmr_ARI"] = gmr_ARI
        output_data["gmr_ll"] = gmr_ll
        output_data["contam_label"] = contaminated_label
        output_data["byzantine_machine"] = byzantine_machine_index

        # ------------------------------
        # GMR + oracle weights
        # ------------------------------
        oracle_trimmed_weights = np.array([
            local_weights[i][j] for i in range(n_split) for j in range(K)
            if i not in byzantine_machine_index[j]
        ])
        oracle_trimmed_weights /= oracle_trimmed_weights.sum()
        oracle_gmr_optimal = np.Inf
        for i in range(n_split):
            oracle_temp = GMR_CTD(
                np.stack([
                    local_means[i][j] for i in range(n_split) for j in range(K)
                    if i not in byzantine_machine_index[j]
                ]),
                np.stack([
                    local_covs[i][j] for i in range(n_split) for j in range(K)
                    if i not in byzantine_machine_index[j]
                ]),
                oracle_trimmed_weights,
                K,
                ground_distance=ground_distance,
                init_method="user",
                means_init=local_means[i],
                covs_init=local_covs[i],
                weights_init=local_weights[i],
            )
            oracle_temp.iterative()
            if oracle_temp.obj < oracle_gmr_optimal:
                oracle_gmr_optimal = oracle_temp.obj
                oracle = oracle_temp
        # oracle_time = time.time() - start_time
        oracle_means, oracle_covs, oracle_weights = (
            oracle.reduced_means,
            oracle.reduced_covs,
            oracle.reduced_weights,
        )
        oracle2true_W1 = GMM_CTD(
            [oracle_means, true_means],
            [oracle_covs, true_covs],
            [oracle_weights, true_weights],
            "W1",
            matrix=False,
        )
        oracle_resp, oracle_predicted_label = label_predict(
            oracle_weights,
            oracle_means,
            oracle_covs,
            GMM_sample,
            return_resp=True,
        )

        oracle_ARI = ARI(true_predicted_labels, oracle_predicted_label)
        oracle_ll = np.log(oracle_resp.sum(1)).sum(0)

        # output_data["oracle_time"] = oracle_time
        output_data["oracle2true_W1"] = oracle2true_W1
        output_data["oracle_ARI"] = oracle_ARI
        output_data["oracle"] = (
            oracle_means,
            oracle_covs,
            oracle_weights,
        )
        output_data["oracle_ll"] = oracle_ll

        for key, item in output_data.items():
            if "W1" in key and "local" not in key:
                print(key, item)

        save_file = os.path.join(
            save_folder,
            "case_" + str(random_state) + "_nsplit_" + str(n_split) +
            "_ncomp_" + str(K) + "_d_" + str(D) + "_ss_" + str(sample_size) +
            "_failurerate_" + str(failure_rate) + "_attackmode_" +
            str(attack_mode) + "_failuretype_" + str(failure_type) + ".pickle",
        )

        f = open(save_file, "wb")
        pickle.dump(output_data, f)
        f.close()


def main(seed, sample_size, overlap, attack_mode, n_split, failure_type):
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
        seed,
        sample_size,
        true_means,
        true_covs,
        true_weights,
        save_folder,
        attack_mode,
        n_split,
        failure_type,
    ]
    Simulation(params)


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
                        default=0.1,
                        help="degree of overlap")
    parser.add_argument(
        "--attack_mode",
        type=int,
        default=1,
        help="1:mean, 2:cov, 3:weight",
    )
    parser.add_argument(
        "--failure_type",
        type=str,
        default="machine",
        help="component or machine",
    )

    args = parser.parse_args()
    sample_size = int(args.ss)
    seed = args.seed
    overlap = args.overlap
    attack_mode = args.attack_mode
    n_split = args.n_split
    failure_type = args.failure_type

    save_folder = os.path.join("./output/save_data/", "ss_" + str(sample_size),
                               "overlap_" + str(overlap))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    main(seed, sample_size, overlap, attack_mode, n_split, failure_type)
