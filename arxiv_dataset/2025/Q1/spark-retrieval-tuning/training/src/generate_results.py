from splits import TPCDS_TEST, TPCH_TEST
import pandas as pd

results = pd.DataFrame()

trials = pd.read_csv("data/trials.csv")

results['query'] = trials[trials['applicationName'].apply(lambda appName: any([appName.startswith("tpch") and appName.endswith(q) for q in TPCH_TEST] + [appName.startswith("tpcds") and appName.endswith(q) for q in TPCDS_TEST]))]['applicationName']
results = results.drop_duplicates(subset=['query']).reset_index(drop=True)

# best runtimes
best_runtimes = trials[['applicationName', 'value']].loc[trials.groupby('applicationName')['value'].idxmin()]
best_runtimes = best_runtimes.rename(columns={"applicationName": "query", "value": "best_runtime"})

results = pd.merge(results, best_runtimes, on='query', how='inner')


# default runtimes
initial_values = trials.groupby('applicationName').first().reset_index()[['applicationName', 'value']]
initial_values = initial_values.rename(columns={"applicationName": "query", "value": "default_runtime"})

results = pd.merge(results, initial_values, on='query', how='inner')


#simtune runtimes
simtune_runtimes = pd.read_csv("src/simtune/results.csv", header=0)
simtune_runtimes = simtune_runtimes.rename(columns={"source_query": "query"})

results = pd.merge(results, simtune_runtimes, on='query', how='inner')


# default runtime on a 3 node docker swarm cluster
default_cluster_results = pd.read_csv("data/default_results_cluster.csv", header=0)
results = pd.merge(results, default_cluster_results, on='query', how='inner')








results.to_csv("final_results.csv", header=True, index=False)

print(results)