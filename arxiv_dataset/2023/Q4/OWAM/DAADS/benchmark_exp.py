import pandas as pd
# import multiprocessing as mp
import multiprocess as mp #CHANGE
import pathlib

from tools.evaluate import aggregate_dataframe, test_then_train

# N_PROCESSES = 3
N_PROCESSES = 1 # CHANGE
DATASETS = ["covertype", "creditcard", "shuttle"]
MODELS = ["AE", "DAE","PW-AE"]
# SEEDS = range(42, 52)
SEEDS = range(42, 44) # CHANGE 
# SUBSAMPLE = 500_000
SUBSAMPLE = 5000 # CHANGE
SAVE_STR = "Benchmark"

CONFIGS = {
    "AE": {"lr": 0.02, "latent_dim": 0.1},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "OC-SVM": {},
    "HST": {"n_trees": 25, "height": 15},
}


perf_metrics, scores = test_then_train(dataset='covertype',
            model='AE',
            seed=42,
            subsample=SUBSAMPLE,
            **CONFIGS.get('AE', {}))

# pool = mp.Pool(processes=N_PROCESSES)
# runs = [
#     pool.apply_async(
#         test_then_train,
#         kwds=dict(
#             dataset=dataset,
#             model=model,
#             seed=seed,
#             subsample=SUBSAMPLE,
#             **CONFIGS.get(model, {}),
#         ),
#     )
#     for dataset in DATASETS
#     for model in MODELS
#     for seed in SEEDS
# ]

# metrics = [run.get()[0] for run in runs]

# metrics_raw = pd.DataFrame(metrics)
# metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

# path = pathlib.Path(__file__).parent.parent.resolve()
# metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
# metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
