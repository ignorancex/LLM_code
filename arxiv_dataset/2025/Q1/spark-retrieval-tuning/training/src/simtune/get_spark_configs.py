import pandas as pd

from ..splits import get_data

test_neighbours = pd.read_csv("src/simtune/neighbours_test.csv", header=0)
trials = get_data("trials", "csv")
trials = trials.dropna()

best_trials = trials.groupby("applicationName")['value'].idxmin()
best_trials = trials.loc[best_trials]

def recommend_spark_configs(trials: pd.DataFrame, neighbours: pd.DataFrame):
    neighbours_with_trials = neighbours.merge(trials, left_on='neighbour_query', right_on='applicationName')

    # keep only the first row of each application because we want to compare against the default parameters
    neighbours_with_trials = neighbours_with_trials.sort_values("source_query_id").groupby('source_query').first().reset_index()
    return neighbours_with_trials

recommend_spark_configs(best_trials, test_neighbours).to_csv("src/simtune/test_spark_configs.csv", index=False)