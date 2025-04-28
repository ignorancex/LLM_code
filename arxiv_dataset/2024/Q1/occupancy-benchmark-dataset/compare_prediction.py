from datetime import date, timedelta
import numpy as np
from scipy import io
from typing import List

from utilities.disturbance_predictor import DisturbancePredictor
from utilities.state_space_gp import QuasiPeriodicKernel
from utilities.kalman_filter import KalmanFilter


# public holidays in Baden-Wuerttemberg, Germany for the year 2019
Holidays = [
    date(2019, 1, 1),
    date(2019, 1, 6),
    date(2019, 4, 19),
    date(2019, 4, 22),
    date(2019, 5, 1),
    date(2019, 5, 30),
    date(2019, 6, 10),
    date(2019, 6, 20),
    date(2019, 10, 3),
    date(2019, 11, 1),
    date(2019, 12, 25),
    date(2019, 12, 26),
]


def compare_prediction(occupancy: np.ndarray, time: np.ndarray, predictor: List[DisturbancePredictor], horizon: int):
    t0 = date(2019, 1, 1)  # data starts at january the 1st, 2019
    RMSE = np.zeros((len(time),))
    RMSE_no_prediction = np.zeros((len(time),))
    for i, t in enumerate(time):
        current_date = t0 + timedelta(seconds=t)
        if current_date in Holidays:
            weekday = 6
        else:
            weekday = current_date.weekday()
        if weekday < 5:
            prediction = predictor[weekday].get_prediction(occupancy[i], horizon)
            RMSE[i] = np.sqrt(np.mean((occupancy[i:horizon + i] - prediction) ** 2))
            RMSE_no_prediction[i] = np.sqrt(np.mean((occupancy[i:horizon + i]) ** 2))

    return RMSE, RMSE_no_prediction


def init_predictor(hyperparameters, Room, dt):
    h = hyperparameters["hyperparameters"][:, :, Room]
    R = hyperparameters["likelihood"][:, Room]
    J = 10
    nu = 1/2
    return [DisturbancePredictor(KalmanFilter, QuasiPeriodicKernel, [J, nu, *h[i, :]], R[i], dt)
            for i in range(5)]


if __name__ == '__main__':
    data = io.loadmat("data/Occupancy.mat")
    Occ = data["Occ"]
    time = data["tspan"].flatten().astype(float)

    dt = time[1]-time[0]
    Rooms = [0, 5, 6, 7, 8]
    Zones = [9, 21, 22, 23, 24]
    steps_year = int(86400*365/dt)  # compare the occupancy for a whole year
    horizon = int(3600*6/dt)  # prediction horizon

    # The hyperparameters for the LFM were trainend with the occupancy data from 01.01.2020 until 31.03.2020.
    hyperparameters = io.loadmat("data/Hyperparameters.mat")
    for i in range(5):
        predictors = init_predictor(hyperparameters, Rooms[i], dt)
        RMSE, RMSE_no_prediction = compare_prediction(occupancy=Occ[0:steps_year + 1 + horizon, Rooms[i]],
                                                      time=time[0:steps_year],
                                                      predictor=predictors,
                                                      horizon=horizon)
        print(f"Zone {Zones[i]}:\n"
              f"RMSE without prediction: {np.sum(RMSE_no_prediction)}\n"
              f"RMSE with prediction: {np.sum(RMSE)}")
