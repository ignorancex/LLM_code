import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import optuna


def create_set(kt, wl_features, expand: bool = False):
    left = wl_features.loc[kt["applicationName_x"]].reset_index(drop=True)
    right = wl_features.loc[kt["applicationName_y"]].reset_index(drop=True)

    if expand:
        features = pd.concat(
            [pd.concat([left, right], axis=1), pd.concat([right, left], axis=1)],
            axis=0,
            ignore_index=True,
        )
        targets = pd.concat([kt["distance"], kt["distance"]], axis=0).reset_index(
            drop=True
        )
    else:
        features = pd.concat([left, right], axis=1)
        targets = kt["distance"].reset_index(drop=True)

    features.columns = left.columns.append(left.columns + "_y")
    return (features, targets)


def objective(trial, data, target, valid_data, valid_target):
    train_x, test_x, train_y, test_y = (
        data,
        valid_data,
        target,
        valid_target,
    )

    param = {
        "metric": "rmse",
        "random_state": 42,
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 500]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.5, 0.7, 1.0]
        ),
        "subsample": trial.suggest_categorical("subsample", [0.4, 0.6, 0.8, 1.0]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.02, 0.05, 0.1]),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, -1]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [10, 20]),
        "cat_smooth": trial.suggest_int("cat_smooth", 1, 100),
        "min_child_samples": None,
        "force_row_wise": True,
    }

    model = LGBMRegressor(**param)
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)])
    preds = model.predict(test_x)
    rmse = mean_squared_error(test_y, preds, squared=False)

    return rmse


# Number of finished trials: 40
# Best trial: {'n_estimators': 100, 'colsample_bytree': 1.0, 'subsample': 0.6, 'learning_rate': 0.05, 'max_depth': -1, 'num_leaves': 33, 'min_data_in_leaf': 20, 'cat_smooth': 31}
# [LightGBM] [Info] Total Bins 19632
# [LightGBM] [Info] Number of data points in the train set: 111222, number of used features: 110
# [LightGBM] [Info] Start training from score 0.474136
# Training RMSE:  0.052151195772381344
# Validation RMSE:  0.0572333831938175
# Test RMSE:  0.057760605285400335
# Now training on the whole train dataset
# [LightGBM] [Info] Total Bins 19782
# [LightGBM] [Info] Number of data points in the train set: 138012, number of used features: 110
# [LightGBM] [Info] Start training from score 0.474340
# Training RMSE:  0.053042574894832314

if __name__ == "__main__":
    base_path = "src/simlearn"

    wl_features = pd.read_csv(
        f"{base_path}/out/workloads.csv", index_col="applicationName"
    )
    kt = pd.read_csv(f"{base_path}/out/kendall_tau_distances.csv")

    train_wl_index, test_wl_index = train_test_split(
        wl_features.index, test_size=0.1, random_state=0
    )

    test_mask = (
        kt[["applicationName_x", "applicationName_y"]].isin(test_wl_index).any(axis=1)
    )
    training_set = kt[~test_mask]
    test_set = kt[test_mask]

    valid_wl_index, test_wl_index = train_test_split(
        test_wl_index, test_size=0.1, random_state=0
    )

    test_mask = (
        test_set[["applicationName_x", "applicationName_y"]]
        .isin(test_wl_index)
        .any(axis=1)
    )
    valid_set = test_set[~test_mask]
    test_set = test_set[test_mask]

    features, targets = create_set(training_set, wl_features, expand=True)
    test_features, test_targets = create_set(test_set, wl_features)
    valid_features, valid_targets = create_set(valid_set, wl_features)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, features, targets, valid_features, valid_targets
        ),
        n_trials=40,
    )
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)

    params = study.best_params
    params["min_child_samples"] = None
    params["force_row_wise"] = True

    model = LGBMRegressor(**params)

    # model = LGBMRegressor(metric="rmse")
    # print(model.get_params())

    model.fit(features, targets)

    train_prediction = model.predict(features)
    test_prediction = model.predict(test_features)
    valid_prediction = model.predict(valid_features)

    print("Training RMSE: ", np.sqrt(mean_squared_error(targets, train_prediction)))
    print(
        "Validation RMSE: ",
        np.sqrt(mean_squared_error(valid_targets, valid_prediction)),
    )
    print("Test RMSE: ", np.sqrt(mean_squared_error(test_targets, test_prediction)))

    print("Now training on the whole train dataset")
    model = LGBMRegressor(**params)

    features, targets = create_set(kt, wl_features, expand=True)
    model.fit(features, targets)

    with open(f"{base_path}/sim_model.pkl", "wb") as file:
        pickle.dump(model, file)

    train_prediction = model.predict(features)
    print("Training RMSE: ", np.sqrt(mean_squared_error(targets, train_prediction)))
