import argparse
import logging
from collections import Counter
from pathlib import Path
from pickle import dump

import pandas as pd
import seaborn as sns

from beck_fiala_column_gen import beck_fiala_column_gen
from contaminated_estimation import get_background, get_pool, contaminated_estimation, calculate_qis, \
    _get_feature_values, calculate_pis
from project_typing import *


def _remove_feature(people: List[Agent], feature: Feature) -> List[Agent]:
    """Drop a feature from all given agents."""
    new_people = []
    for person in people:
        new_person = person.copy()
        del new_person[feature]
        new_people.append(new_person)
    return new_people


def run(positives: List[Agent], directory: Path, pool_pis: List[float], k: int):
    pool_distribution = beck_fiala_column_gen(pool_pis, positives)
    with open(directory.joinpath("pool_distribution.pickle"), "wb") as file:
        dump({"pool_pis": pool_pis, "k": k, "pool_distribution": pool_distribution}, file)
    with open(directory.joinpath("pool_distribution.txt"), "w") as file:
        file.write("pool_pis:\n")
        file.write(str(pool_pis) + "\n")
        file.write(f"k={k}\n")
        file.write("Distribution:")
        for panel, prob in pool_distribution:
            file.write(f"{str(panel)}\t{prob}\n")

    fv_representation_dist: Dict[FeatureValue, Dict[int, float]] = {}
    for panel, prob in pool_distribution:
        counter = Counter()
        for i in panel:
            counter.update(list(positives[i].items()))
        for fv, frequency in counter.items():
            if fv not in fv_representation_dist:
                fv_representation_dist[fv] = {}
            if frequency not in fv_representation_dist[fv]:
                fv_representation_dist[fv][frequency] = 0.
            fv_representation_dist[fv][frequency] += prob
    data = []
    for fv in fv_representation_dist:
        for i in range(k + 1):
            if i in fv_representation_dist[fv]:
                prob = fv_representation_dist[fv][i]
            else:
                prob = 0.
            data.append({"feature-value pair": fv, "number of panel members": i, "probability": prob})
    with open(directory.joinpath("histogram_data.pickle"), "wb") as file:
        dump({"fv_representation_dist": fv_representation_dist, "data": data}, file)
    df = pd.DataFrame(data)
    f = sns.catplot(data=df, x="number of panel members", y="probability", row="feature-value pair", kind="bar")
    f.savefig(directory.joinpath("histograms.pdf"))


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Estimate participation probabilities and run selection algorithm on pool.")
    parser.add_argument("directory", type=str, help="save all generated data in ./output/{directory}/")
    parser.add_argument("--dropother", default=False, action="store_true",
                        help="drop agents with nonbinary gender from pool for consistency with ESS")
    parser.add_argument("--dropclimate", default=False, action="store_true",
                        help="drop climate concern feature to prevent very low participation probabilities")
    parser.add_argument("--run", default=False, action="store_true",
                        help="run selection algorithm on pool (might take some time)")
    parser.add_argument("--householdsize", type=float, default=2.,
                        help="average number of eligible agents per household")
    args = parser.parse_args()

    # Set up output
    directory_name = "./output/" + args.directory
    directory = Path(directory_name)
    directory.mkdir(parents=False, exist_ok=True)
    est_dir = directory.joinpath("estimation")
    est_dir.mkdir(parents=False, exist_ok=True)
    if args.run:
        run_dir = directory.joinpath("algorithm")
        run_dir.mkdir(parents=False, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S",
                        filename=directory.joinpath("pool.log"),
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Data preprocessing
    background, weights = get_background()
    pool = get_pool()
    if set(background[0].keys()) != set(pool[0].keys()):
        raise ValueError(f"Mismatch of attributes. Background: {background[0].keys()}, Positives: {pool[0].keys()}")
    pi = len(pool) / 30000 / args.householdsize
    if args.dropother:
        logging.info("Drop agents with gender 'other'.")
        pool = [person for person in pool if person["gender"] != "Other"]
        background = [person for person in background if person["gender"] != "Other"]
    if args.dropclimate:
        logging.info("Drop feature 'climate_concern_level'.")
        climate_concern = Feature("climate_concern_level")
        pool = _remove_feature(pool, climate_concern)
        background = _remove_feature(background, climate_concern)
    feature_values = _get_feature_values(background + pool)

    # Learn the β values
    baseline, value_multipliers = contaminated_estimation(pool, background, pi, weights)
    with open(est_dir.joinpath("betas.pickle"), "wb") as file:
        dump({"baseline": baseline, "value_multipliers": value_multipliers}, file)
    fv_s = [(value_multipliers[fv], fv) for fv in value_multipliers]  # sort by magnitude of β_{f,v}
    fv_s.sort()
    logging.info(f"Baseline β₀ =\t{baseline:.2%}")
    for mult, (feature, value) in fv_s:
        logging.info(f"β_({feature},{value}) =\t{mult:.2%}")

    k = 110

    # Compute the statistics q* and α
    qstar = baseline
    for feature, values in feature_values.items():
        qstar *= min(value_multipliers[(feature, value)] for value in values)
    alpha = qstar * 60000 / k
    logging.info(f"q* = {qstar},\tα = {alpha}.")

    # Calculate participation probabilities qᵢ for agents in pool and background sample
    pool_qis = calculate_qis(pool, baseline, value_multipliers)
    df = pd.DataFrame(pool)
    df["qi"] = pool_qis
    df.to_csv(est_dir.joinpath("pool_qis.csv"))
    background_qis = calculate_qis(background, baseline, value_multipliers)
    df = pd.DataFrame(background)
    df["weight"] = weights
    df["qi"] = background_qis
    df.to_csv(est_dir.joinpath("background_qis.csv"))
    with open(est_dir.joinpath("background_qis.pickle"), "wb") as file:
        dump({"background": background, "feature_values": feature_values, "background_qis": background_qis,
              "background_weights": weights}, file)

    # Calculate the marginal selection probabilities and check whether the pool was good
    pool_pis, num_deterministic = calculate_pis(pool, pool_qis, k)

    good_pool = True
    if num_deterministic > 0:
        logging.info(f"Pool not good because {num_deterministic} unclipped πs would exceed 1.")
        good_pool = False
    for feature, values in feature_values.items():
        for value in values:
            pop_frac = sum(weight for person, weight in zip(background, weights) if person[feature] == value) / sum(weights)
            sum_pis = sum(pi for person, pi in zip(pool, pool_pis) if person[feature] == value)
            if (1 - alpha**(-.49)) * k * pop_frac > sum_pis:
                logging.info(f"Pool not good: Σ π_i,P for {feature}:{value} below (1 - {alpha}^(-.49) * k * nfv / n.")
                good_pool = False
            elif (1 + alpha**(-.49)) * k * pop_frac < sum_pis:
                logging.info(f"Pool not good: Σ π_i,P for {feature}:{value} exceeds (1 + {alpha}^(-.49) * k * nfv / n.")
                good_pool = False
    sum_ais = sum(1/qi for qi in pool_qis)
    if sum_ais > 60000 / (1 - alpha**(-.49)):
        logging.info("Pool not good because Σ aᵢ too large.")
        good_pool = False
    if good_pool:
        logging.info("Pool is good!")

    with open(est_dir.joinpath("pool_pis.pickle"), "wb") as file:
        dump({"pool_qis": pool_qis, "pool_pis": pool_pis, "num_deterministic": num_deterministic, "k": k, "q*": qstar,
              "α": alpha, "good": good_pool}, file)

    # compare the qᵢ-weighted prevalence of feature intersections in the background sample with pool
    fvs = [(feature, value) for feature in feature_values for value in feature_values[feature]]
    background_normalization = sum(weight * qi for weight, qi in zip(weights, background_qis))
    pair_fraction_comp: Dict[Tuple[FeatureValue, FeatureValue], Tuple[float, float]] = {}
    for feature1, value1 in fvs:
        for feature2, value2 in fvs:
            if feature1 <= feature2:
                continue
            pool_fraction = sum(
                int((person[feature1], person[feature2]) == (value1, value2)) for person in pool) / len(pool)
            background_numerator = sum(int((person[feature1], person[feature2]) == (value1, value2)) * weight * qi
                                       for person, weight, qi in zip(background, weights, background_qis))
            background_fraction = background_numerator / background_normalization
            logging.debug(f"{feature1}:{value1} ∩ {feature2}:{value2}: in pool {pool_fraction:.2%}, qᵢ-weighted in "
                          f"background {background_fraction:.2%}")
            pair_fraction_comp[((feature1, value1), (feature2, value2))] = pool_fraction, background_fraction
    l = [(x - y, f1, v1, f2, v2, x, y) for ((f1, v1), (f2, v2)), (x, y) in pair_fraction_comp.items()]
    l.sort()  # sort by difference between pool fraction
    df = pd.DataFrame(l, columns=["difference", "feature1", "value1", "feature2", "value2", "density in pool",
                                   "qᵢ-weighted density in background"])
    del df["difference"]
    df.to_csv(est_dir.joinpath("pairwise_correlations.csv"))
    f = sns.relplot(data=df, x="density in pool", y="qᵢ-weighted density in background")
    f.savefig(est_dir.joinpath("pairwise_correlations.pdf"))

    if args.run:
        logging.info("Start running the sampling algorithm.")
        run(pool, run_dir, pool_pis, k)


if __name__ == "__main__":
    main()
