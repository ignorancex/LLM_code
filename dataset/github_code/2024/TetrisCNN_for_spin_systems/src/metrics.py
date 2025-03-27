import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def reshape_by_time(in_tensor, time_tab):
    if type(in_tensor) != np.ndarray:
        in_tensor = in_tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(in_tensor)
    tensor_df["time"] = time_tab
    tensor_avg = tensor_df.groupby("time").mean().to_numpy()
    tensor_std = tensor_df.groupby("time").std().to_numpy()
    return tensor_avg, tensor_std


def calculate_r2_and_mse(
    truth,
    answer,
    time_tab=None,
    exp=None,
    old_way=False,
):
    if time_tab is None and exp is None:
        old_way = True

    if not old_way:
        truth = truth.reshape(-1, 1) if len(truth.shape) < 2 else truth
        answer = answer.reshape(-1, 1) if len(answer.shape) < 2 else answer

        if exp.y_scaler is not None:
            truth_rescaled = np.array(
                [
                    exp.y_scaler[i]
                    .inverse_transform(truth[:, i].reshape(-1, 1))
                    .reshape(-1)
                    for i in range(truth.shape[1])
                ]
            ).T
            answer_rescaled = np.array(
                [
                    exp.y_scaler[i]
                    .inverse_transform(answer[:, i].reshape(-1, 1))
                    .reshape(-1)
                    for i in range(answer.shape[1])
                ]
            ).T
        else:
            truth_rescaled = truth
            answer_rescaled = answer

        truth_avg, _ = reshape_by_time(truth_rescaled, time_tab)
        answer_avg, _ = reshape_by_time(answer_rescaled, time_tab)

        r2 = r2_score(truth_avg, answer_avg, multioutput="variance_weighted")
        mse = mean_squared_error(truth_avg, answer_avg, multioutput="uniform_average")
    else:
        r2 = r2_score(truth, answer)
        mse = mean_squared_error(truth, answer)

    return r2, mse
