import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from t1dsim_ai.options import states, inputs, input_ind


def digitalTwin_scenario(
    meal_size_array=[75],  # g
    meal_time_fromStart_array=[60],  # min
    init_cgm=110,  # mg/dL
    basal_insulin=1,  # U/h
    carb_ratio=12,
    sim_time=5 * 60,
    hr=70,  # int or array of len sim_time
    initial_time="08:00:00",
    bedtime=13 * 60,  # Bedtime since start simulation
    sleep_duration=8,  # Sleep duration in hours
    exercise_time=60 * 2,
    exercise_duration=0.5,
):
    np.random.seed(0)
    dfInitStates = pd.read_csv(
        Path(__file__).parent / "models/initSteadyStates.csv"
    ).set_index("initCGM")

    base_date = datetime.datetime(2024, 8, 15)
    (h, m, s) = initial_time.split(":")
    initial_time = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))

    df_scenario = pd.DataFrame()
    df_scenario["time"] = pd.date_range(
        pd.Timestamp(base_date + initial_time),
        pd.Timestamp(base_date + initial_time + datetime.timedelta(minutes=sim_time)),
        freq="5 min",
    )

    df_scenario[states + inputs + input_ind] = 0.0
    df_scenario["feat_hour_of_day_cos"] = np.cos(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario["feat_hour_of_day_cos"] = np.sin(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario.loc[0, "cgm"] = init_cgm
    df_scenario.loc[
        np.array(meal_time_fromStart_array) // 5, "input_meal_carbs"
    ] = meal_size_array

    df_scenario["input_insulin"] = basal_insulin
    df_scenario.loc[np.array(meal_time_fromStart_array) // 5, "input_insulin"] = (
        12 * np.array(meal_size_array) / carb_ratio
    )

    df_scenario["heart_rate"] = hr + np.random.normal(0, 2, len(df_scenario))

    df_scenario.loc[0, states] = dfInitStates.loc[int(init_cgm), states]

    df_scenario.loc[
        bedtime // 5 : (bedtime + sleep_duration * 60) // 5, "sleep_efficiency"
    ] = 1
    df_scenario.loc[
        bedtime // 5 : (bedtime + sleep_duration * 60) // 5, "heart_rate"
    ] = (hr - 10 + np.random.normal(0, 1, sleep_duration * 12 + 1))

    # df_scenario.loc[
    #    exercise_time // 5 : (exercise_time + exercise_duration * 60) // 5, "heart_rate"
    # ] = (hr + 30 + np.random.normal(0, 1, int(exercise_duration * 12) + 1))
    df_scenario["heart_rate_WRTbaseline"] = df_scenario["heart_rate"] - hr

    return df_scenario
