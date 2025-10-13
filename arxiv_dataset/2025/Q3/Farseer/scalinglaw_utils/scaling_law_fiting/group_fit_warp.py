
import math
import pandas as pd
from pandas import DataFrame
from scalinglaw_utils.basic_fitting_tool import LinearFit

def group_fit(df: DataFrame, group_key: str, x_key: str, y_key: str, x_log=True, y_log=True, compute_partial_loss=True, min_len=5, weight=None):
    if (group_key != ''):
        df_c = df.copy().sort_values([group_key, x_key, y_key])
        unique_values = df_c[group_key].unique().tolist()
    else:
        df_c = df.copy().sort_values([x_key, y_key])
        unique_values = ['']
    res = list()
    residuals_df = pd.DataFrame(index=df_c.index)
    prefix = (((((group_key + '_x-') + x_key) + '_y-') + y_key) if (group_key != '') else ((('x-' + x_key) + '_y-') + y_key))
    for unique_value in unique_values:
        group = (df_c[(df_c[group_key] == unique_value)] if (unique_value != '') else df_c)
        X = group[x_key].values
        Y = group[y_key].values
        weights = (group[weight].values if (weight is not None) else None)
        if (len(X) <= min_len):
            continue
        model = LinearFit(X, Y).fit(x_log, y_log, 'sklearn', weights)
        residuals_df.loc[(group.index, (prefix + '_pred_raw'))] = model.pred_raw
        residuals_df.loc[(group.index, (prefix + '_pred_y'))] = model.pred_y
        residuals_df.loc[(group.index, (prefix + '_residual'))] = model.residual
        residuals_df.loc[(group.index, (prefix + '_relative_residual'))] = model.relative_residual
        residuals_df.loc[(group.index, (prefix + '_slope'))] = model.slope
        residuals_df.loc[(group.index, (prefix + '_intercept'))] = model.intercept
        fit_res = [unique_value, model.slope, model.intercept, model.residual_mean, model.relative_residual_mean, model.residual_trans_mean, abs(model.slope), model.error_mean, model.relative_error_mean]
        if compute_partial_loss:
            pred_partial_loss = ((model.pred_raw / (1 - (math.sqrt(2) ** model.slope))) * (math.sqrt(2) ** model.slope))
            residuals_df.loc[(group.index, (prefix + '_pred_partial_loss'))] = pred_partial_loss
            residuals_df.loc[(group.index, (prefix + '_pred_partial_loss_delta'))] = (group.loc[(group.index, 'L')] - pred_partial_loss)
            fit_res = (fit_res + [(math.pow(10, model.intercept) / (1 - (math.sqrt(2) ** model.slope))), float(residuals_df[(prefix + '_pred_partial_loss_delta')].mean())])
        res.append(fit_res)
    df_c = pd.concat([df_c, residuals_df], axis=1)
    columns_name = [group_key, 'slope', 'intercept', 'residual_mean', 'relative_residual_mean', 'residual_trans_mean', 'slope_abs', 'error_mean', 'relative_error_mean']
    if compute_partial_loss:
        columns_name = (columns_name + ['intercept_abs', 'pred_partial_loss_delta_mean'])
    res_df = pd.DataFrame(res, columns=columns_name)
    return (df_c, res_df)

def get_x_log_y_log(j):
    (x_log, y_log) = (False, False)
    if (j == 1):
        (x_log, y_log) = (False, True)
    if (j == 2):
        (x_log, y_log) = (True, False)
    if (j == 3):
        (x_log, y_log) = (True, True)
    return (x_log, y_log)

def multi_fit_warp(df: DataFrame, x_key: (str | list[str]), y_key: (str | list[str]), x_log=True, y_log=True, weight=None):
    x_keys = (x_key if isinstance(x_key, list) else [x_key])
    y_keys = (y_key if isinstance(y_key, list) else [y_key])
    unique_keys = list(set((x_keys + y_keys)))
    df_c = df.copy().sort_values(unique_keys)
    residuals_df = pd.DataFrame(index=df_c.index)
    res = list()
    for xk in x_keys:
        for yk in y_keys:
            prefix = ((('x-' + x_key) + '_y-') + yk)
            X = df_c[xk].to_numpy()
            Y = df_c[yk].to_numpy()
            weights = (df_c[weight].values if (weight is not None) else None)
            model = LinearFit(X, Y).fit(x_log, y_log, 'sklearn', weights)
            residuals_df.loc[(df_c.index, (prefix + '_pred_raw'))] = model.pred_raw
            residuals_df.loc[(df_c.index, (prefix + '_pred_y'))] = model.pred_y
            residuals_df.loc[(df_c.index, (prefix + '_residual'))] = model.residual
            residuals_df.loc[(df_c.index, (prefix + '_relative_residual'))] = model.relative_residual
            residuals_df.loc[(df_c.index, (prefix + '_slope'))] = model.slope
            residuals_df.loc[(df_c.index, (prefix + '_intercept'))] = model.intercept
            fit_res = [prefix, model.slope, model.intercept, model.residual_mean, model.relative_residual_mean, model.residual_trans_mean, abs(model.slope), model.error_mean, model.relative_error_mean]
            res.append(fit_res)
    df_c = pd.concat([df_c, residuals_df], axis=1)
    columns_name = ['name', 'slope', 'intercept', 'residual_mean', 'relative_residual_mean', 'residual_trans_mean', 'slope_abs', 'error_mean', 'relative_error_mean']
    res_df = pd.DataFrame(res, columns=columns_name)
    return (df_c, res_df)
