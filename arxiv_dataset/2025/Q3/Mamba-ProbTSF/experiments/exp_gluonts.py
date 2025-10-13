# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from tqdm import tqdm
import argparse
import gc

from gluonts_utils import original_shape,save_forecast_outputs
from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

# Define parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument('--folder', type=str, default='ECL_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')#'Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')
parser.add_argument('--prediction_length', type=int, default=96)
parser.add_argument('--past_length', type=int, default=96)
parser.add_argument('--mamba_batch_size', type=int, default=16)
parser.add_argument('--Method', type=str, default='DeepAR')
#save_bounds = True  # <-- Set True to also save lower/upper bounds
parser.add_argument('--save_bounds', type=bool, default=True)


# Parse args
args = parser.parse_args()

# Assign variables
folder = 'results_prob/'+ args.folder
prediction_length = args.prediction_length
past_length = args.past_length
mamba_batch_size = args.mamba_batch_size
Method = args.Method
save_bounds = args.save_bounds


# Now you can use them normally
print(f"Folder: {folder}")
print(f"Past length: {past_length}")
print(f"Prediction length: {prediction_length}")
print(f"Method: {Method}")


# %%
#Loads same training and testing dataset as used in Mamba-ProbTSF, but in gluonts format.
#This requires one to have run the Mamba-ProbTSF, save exactly how Mamba-ProbTSF separated the test and training data and use the same training data.
past = np.load(folder+'/input.npy')
fut  = np.load(folder+'/trues.npy')
test_shape = past.shape

past = np.transpose(past, (0, 2, 1)).reshape(-1,past_length)
fut  = np.transpose(fut , (0, 2, 1)).reshape(-1,prediction_length)

test_dataset = np.hstack((past,fut))
print(test_shape)
del past
del fut

past = np.load(folder+'/train_dataset/train_input.npy')
fut  = np.load(folder+'/train_dataset/train_trues.npy')

past = np.transpose(past, (0, 2, 1)).reshape(-1,past_length)
fut  = np.transpose(fut , (0, 2, 1)).reshape(-1,prediction_length)

custom_dataset = np.hstack((past,fut))
print(custom_dataset.shape)
del past
del fut

# if 'Traffic' in folder:
#     print(custom_dataset.shape)
#     ind = np.linspace(0,len(custom_dataset)-1,7500000).astype(int)
#     custom_dataset = custom_dataset[ind]

print(test_dataset.shape,custom_dataset.shape)

# %%

def process_single_forecast(test_sample, predictor, prediction_length, quantiles, save_bounds=save_bounds):
    """
    Process a single time series forecast and return forecast stats.
    """

    #forecast_it, ts_it = test_sample
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_sample,
        predictor=predictor,
        num_samples=1000,
    )
    forecast = list(forecast_it)[0]
    ts = list(ts_it)[0].to_numpy().reshape(-1)

    past, fut = ts[:-prediction_length], ts[-prediction_length:]

    forecast_means = forecast.mean
    forecast_medians = forecast.median
    forecast_futures = fut

    forecast_within_intervals = []
    forecast_lower_bounds = []
    forecast_upper_bounds = []

    for k in range(3):
        lower = forecast.quantile(quantiles[k, 0])
        upper = forecast.quantile(quantiles[k, 1])
        within = np.logical_and(fut > lower, fut < upper)
        forecast_within_intervals.append(within)

        if save_bounds:
            forecast_lower_bounds.append(lower)
            forecast_upper_bounds.append(upper)

    return (forecast_means, forecast_medians, forecast_futures, 
            forecast_within_intervals, forecast_lower_bounds, forecast_upper_bounds)

perbatch = test_shape[-1]
batch_size = mamba_batch_size * perbatch
print(f"Batch size: {batch_size}")
# %%
freq = "1H"
start = pd.Period("01-01-2019", freq=freq)  # Just part of gluonts test, nothing to be attributed from this

# %%
# train dataset: 
train_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset],
    freq=freq,
)

# test dataset: 
test_ds = ListDataset(
    [{"target": x, "start": start} for x in test_dataset], freq=freq
)

if Method == 'DeepAR':
    from gluonts.torch.model.deepar import DeepAREstimator
    estimator = DeepAREstimator(
        freq=freq,
        batch_size=perbatch,
        prediction_length=prediction_length,
        context_length=past_length,
        trainer_kwargs=dict(
            max_epochs=20,
            accelerator="gpu",  # <- use "cpu" if needed
            devices=1
        ),
    )
    predictor = estimator.train(train_ds)

del custom_dataset
del train_ds

# --- Configuration ---
confidence = np.array([0.683, 0.955, 0.998])
quantiles = (1 + np.array([[-1], [1]]) * confidence).T / 2  # shape (3,2)

# Output folder
folder_path = os.path.join(folder, 'gluonts', Method)
os.makedirs(folder_path, exist_ok=True)

# --- Main loop ---
means = np.empty((batch_size//4,prediction_length))#[]
medians = np.empty_like(means)#[]
futures = np.empty_like(means)
within_intervals = [np.empty_like(means) for _ in range(3)]
lower_bounds = [np.empty_like(means) for _ in range(3)] if save_bounds else None
upper_bounds = [np.empty_like(means) for _ in range(3)] if save_bounds else None

# forecast_it, ts_it = make_evaluation_predictions(
#     dataset=test_ds,
#     predictor=predictor,
#     num_samples=100,
# )

# print('startingthis')
# # Convert iterators to lists
# forecasts = list(forecast_it)
# tss = list(ts_it)
# print('endingthis')

for idx in tqdm(range(batch_size//4)):
    gc.collect()

    (mean_val, median_val, future_val, within_vals, lower_vals, upper_vals) = process_single_forecast(
        test_ds[idx:idx+1], predictor, prediction_length, quantiles, save_bounds=save_bounds
    )
    # (mean_val, median_val, future_val, within_vals, lower_vals, upper_vals) = process_single_forecast(
    #     [forecasts[idx],tss[idx]], predictor, prediction_length, quantiles, save_bounds=save_bounds
    # )

    means[idx] = mean_val
    medians[idx] = median_val
    futures[idx] = future_val
    for k in range(3):
        within_intervals[k][idx] = within_vals[k]

        if save_bounds:
            lower_bounds[k][idx] = lower_vals[k]
            upper_bounds[k][idx] = upper_vals[k]

    if (idx+1)%test_shape[-1]==0:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds,original_shape=lambda x: original_shape(x[:idx+1],test_shape))

    elif (idx+1)%50==0 and idx<test_shape[-1]:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds,original_shape=lambda x:x[:idx+1])

save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds, save_bounds,original_shape=lambda x: original_shape(x[:idx+1],test_shape))

del test_dataset
del test_ds
