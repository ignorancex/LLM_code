import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

HYPO = 70.0
HYPER = 180.0
L_BOUND = 20.0
U_BOUND = 420.0


def shift_feature(cgm, horizon, col_name, shift_tolerance):
    cgm2 = cgm.clone().select(["date", "mg/dl"])
    cgm2 = cgm2.rename({"mg/dl": col_name})

    cgm = cgm.with_columns(
        (pl.col("date") - pl.duration(minutes=horizon)).alias("dateShift")
    )
    cgm = cgm.join_asof(
        cgm2, left_on="dateShift", right_on="date", tolerance=shift_tolerance
    )
    return cgm.drop(["dateShift", "date_right"])


def load_transform(data_path, subject_id, horizons, shift_tolerance):
    # Read data
    cgm = pl.read_excel(f"{data_path}/Subject{subject_id}.xlsx", sheet_name="CGM")
    cgm = cgm.unique(subset="date").sort("date")

    # Generate sequence
    for horizon in horizons:
        cgm = shift_feature(cgm, horizon, f"horizon{abs(horizon)}", shift_tolerance)

    cgm = cgm.drop("mg/dl")
    df = cgm.to_pandas()

    # Assign labels
    df["bgClass"] = df[f"horizon{abs(horizons[-1])}"].apply(
        lambda x: (
            np.nan
            if pd.isna(x)
            else ("Hypo" if x < HYPO else "Hyper" if x > HYPER else "Normal")
        )
    )
    df["subject_id"] = subject_id
    df = df.dropna()

    return df.copy()


def get_data(
    data_path,
    subjects,
    horizons,
    train_split,
    test_split,
    single_output=False,
    scale=True,
    shift_tolerance="1m",
):
    df = pd.DataFrame()
    for subject in subjects:
        sub_df = load_transform(data_path, subject, horizons, shift_tolerance)
        df = sub_df if df.empty else pd.concat([df, sub_df])
    df = df.reset_index(drop=True)

    input_cols = [col for col in df.columns if "horizon" in col]

    if scale:
        df = scale_data(df, [col for col in input_cols if "horizon" in col])

    # Get train & test set
    train_set = df[df["subject_id"].isin(train_split)].reset_index(drop=True)
    test_set = df[df["subject_id"].isin(test_split)].reset_index(drop=True)
    print(f"Train size\t{len(train_set)}")
    print(f"Test size\t{len(test_set)}")
    print("\n")

    # Get input & output column names
    X_cols = input_cols[: int(len(input_cols) / 2)]
    y_cols = (
        [input_cols[-1]] if single_output else input_cols[int(len(input_cols) / 2) :]
    )
    print(f"Feature columns\t{X_cols}")
    print(f"Target columns\t{y_cols}")
    print("\n")

    return train_set, test_set, X_cols, y_cols


def scale_data(df, scale_cols):
    for col in scale_cols:
        df[col] = (2 * (df[col] - L_BOUND) / (U_BOUND - L_BOUND)) - 1
    return df


def rescale_data(df, rescale_cols):
    for col in rescale_cols:
        df[col] = ((df[col] + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    return df


def print_results(df):
    print("Cumulative")
    samples = 0
    maes, mapes, rmses = [], [], []
    for subject in df["subject_id"].unique():
        x = df[df["subject_id"] == subject]
        samples += len(x)
        maes.append(mean_absolute_error(x["target"], x["y_pred"]))
        mapes.append(mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100)
        rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))
    print(f"Samples: {samples}")
    print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
    print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
    print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")

    for condition in ["Normal", "Hyper", "Hypo"]:
        dummy = df[df["bgClass"] == condition]
        samples = 0
        maes, mapes, rmses = [], [], []
        for subject in df["subject_id"].unique():
            x = dummy[dummy["subject_id"] == subject]
            samples += len(x)
            if x.empty:
                continue
            maes.append(mean_absolute_error(x["target"], x["y_pred"]))
            mapes.append(mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100)
            rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))

        print("~~~~~~~~~~")
        print(f"{condition}")
        print(f"Samples: {samples}")
        print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
        print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
        print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")
