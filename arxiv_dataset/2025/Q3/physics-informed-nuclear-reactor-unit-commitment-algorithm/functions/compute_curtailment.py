import pandas as pd
import numpy as np
from typing import Union, Optional

def compute_curtailment(
    gen_result_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    result: dict,
    *,
    hour_from_columns: bool = True,
    hour_index: Optional[Union[pd.Index, np.ndarray, list]] = None,
    wind_tag: str = "wec_sdge_onshore_wind_turbine_1.0",
    solar_tag: str = "wec_sdge_solar_photovoltaic_1.0",
    wind_resource_name: str = "onshore_wind_turbine",
    solar_resource_name: str = "solar_photovoltaic",
    floor_zero: bool = True
) -> pd.DataFrame:
    """
    Compute curtailment (total and per-technology) from UC outputs.

    Parameters
    ----------
    gen_result_df : DataFrame
        Wide generator output (rows: generators aligned with gen_df; cols: hours).
    gen_df : DataFrame
        Generator metadata with at least ['r_id','resource','existing_cap_mw'].
    result : dict
        Must include key 'gen_var_cf': DataFrame with columns
        ['r_id','hour','cf','existing_cap_mw','gen_full'].
    hour_from_columns : bool, default True
        If True, interpret melted hour from gen_result_df columns and +1.
        If False, use hour_index as the hour labels.
    hour_index : array-like or Index, optional
        Hour labels when hour_from_columns=False.
    wind_tag, solar_tag : str
        Labels in result['gen_var_cf']['gen_full'] for wind/solar CF rows.
    wind_resource_name, solar_resource_name : str
        Labels in gen_df['resource'] for wind/solar capacity rows.
    floor_zero : bool, default True
        Clip any negative curtailment values to 0.

    Returns
    -------
    DataFrame with columns:
        ['hour','cf_wind','cf_solar','curt','max_vre_possible','gen',
         'gen_wind','gen_solar','cap_wind','cap_solar',
         'solar_possible','wind_possible','wind_curtail','solar_curtail']
    """

    # ---- 1) Long-form generation (r_id, hour, gen) ----
    gen_result_df_long = gen_result_df.copy(deep=True)
    gen_result_df_long["r_id"] = gen_df["r_id"].values
    gen_result_df_long = pd.melt(
        gen_result_df_long, id_vars=["r_id"], var_name="hour", value_name="gen"
    )

    if hour_from_columns:
        # columns assumed integer hour indices -> convert to int and make 1-based
        gen_result_df_long["hour"] = gen_result_df_long["hour"].astype(int) + 1
    else:
        if hour_index is None:
            raise ValueError("hour_index must be provided when hour_from_columns=False.")
        # Ensure stable r_id ordering then tile provided hour_index per r_id
        gen_result_df_long.sort_values(["r_id", "hour"], inplace=True)
        rids = gen_result_df_long["r_id"].unique()
        tiled = np.concatenate([np.asarray(hour_index) for _ in rids])
        if len(tiled) != len(gen_result_df_long):
            raise ValueError("hour_index length does not match melted rows per r_id.")
        gen_result_df_long["hour"] = tiled

    # ---- 2) Merge with variable CF table ----
    if "gen_var_cf" not in result:
        raise KeyError("result must contain 'gen_var_cf' DataFrame.")
    gen_var_cf = result["gen_var_cf"]
    required = {"r_id", "hour", "cf", "existing_cap_mw", "gen_full"}
    if not required.issubset(gen_var_cf.columns):
        missing = required - set(gen_var_cf.columns)
        raise KeyError(f"gen_var_cf is missing columns: {missing}")

    curtail = gen_var_cf.merge(gen_result_df_long, how="inner", on=["r_id", "hour"])

    # ---- 3) Base curtailment and max VRE potential (per unit) ----
    curtail["curt"] = curtail["cf"] * curtail["existing_cap_mw"] - curtail["gen"]
    curtail["max_vre_possible"] = curtail["cf"] * curtail["existing_cap_mw"]
    if floor_zero:
        curtail["curt"] = curtail["curt"].clip(lower=0)

    # ---- 4) Pivot CFs (hour x tech), and aggregate hour sums ----
    pivot_cf = curtail.pivot_table(index="hour", columns="gen_full", values="cf", aggfunc="first")

    sums = curtail.groupby("hour", as_index=False).agg(
        curt=("curt", "sum"),
        max_vre_possible=("max_vre_possible", "sum"),
        gen=("gen", "sum"),
    )

    # ---- 5) Per-tech actual generation (wind, solar) ----
    gen_wind = (
        curtail.loc[curtail["gen_full"] == wind_tag]
        .groupby("hour", as_index=False)["gen"].sum()
        .rename(columns={"gen": "gen_wind"})
    )
    gen_solar = (
        curtail.loc[curtail["gen_full"] == solar_tag]
        .groupby("hour", as_index=False)["gen"].sum()
        .rename(columns={"gen": "gen_solar"})
    )

    # ---- 6) Merge hour-level tables ----
    out = pivot_cf.merge(sums, on="hour", how="left")
    out = out.merge(gen_wind, on="hour", how="left")
    out = out.merge(gen_solar, on="hour", how="left")

    # ---- 7) Rename CF columns; ensure presence ----
    if wind_tag in out.columns:
        out = out.rename(columns={wind_tag: "cf_wind"})
    else:
        out["cf_wind"] = 0.0
    if solar_tag in out.columns:
        out = out.rename(columns={solar_tag: "cf_solar"})
    else:
        out["cf_solar"] = 0.0

    # ---- 8) Installed capacities (sum across rows for robustness) ----
    cap_wind = gen_df.loc[gen_df["resource"] == wind_resource_name, "existing_cap_mw"].sum()
    cap_solar = gen_df.loc[gen_df["resource"] == solar_resource_name, "existing_cap_mw"].sum()
    out["cap_wind"] = float(cap_wind) if pd.notnull(cap_wind) else 0.0
    out["cap_solar"] = float(cap_solar) if pd.notnull(cap_solar) else 0.0

    # ---- 9) Tech-specific possibles and curtailments ----
    out["gen_wind"] = out["gen_wind"].fillna(0.0)
    out["gen_solar"] = out["gen_solar"].fillna(0.0)

    out["solar_possible"] = out["cap_solar"] * out["cf_solar"]
    out["wind_possible"]  = out["cap_wind"]  * out["cf_wind"]

    out["wind_curtail"]  = out["wind_possible"]  - out["gen_wind"]
    out["solar_curtail"] = out["solar_possible"] - out["gen_solar"]
    if floor_zero:
        out["wind_curtail"]  = out["wind_curtail"].clip(lower=0)
        out["solar_curtail"] = out["solar_curtail"].clip(lower=0)

    # ---- 10) Column order + final shape ----
    out = out.reset_index()
    desired_cols = [
        "hour", "cf_wind", "cf_solar", "curt", "max_vre_possible",
        "gen", "gen_wind", "gen_solar",
        "cap_wind", "cap_solar",
        "solar_possible", "wind_possible",
        "wind_curtail", "solar_curtail",
    ]
    for c in desired_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[desired_cols]

    return out
