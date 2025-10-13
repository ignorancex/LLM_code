from t1dsim_ai.options import (
    states_name,
    dict_states,
    inputs,
)
import numpy as np
from pickle import dump, load
from sklearn.preprocessing import RobustScaler


def scaler(x_est, u_id, path_scaler, train=False):
    if train:
        scaler_states = RobustScaler()  # MinMaxScaler()
        scaler_inputs = RobustScaler()  # MinMaxScaler()

        x_est = scaler_states.fit_transform(x_est)
        u_id = scaler_inputs.fit_transform(u_id)

        # Save the scaler
        dump(scaler_states, open(path_scaler + "scaler_states.pkl", "wb"))
        dump(scaler_inputs, open(path_scaler + "scaler_inputs.pkl", "wb"))

        return x_est, u_id
    else:
        scaler_states = load(open(path_scaler + "scaler_states.pkl", "rb"))
        scaler_inputs = load(open(path_scaler + "scaler_inputs.pkl", "rb"))

        x_est = scaler_states.transform(x_est)
        u_id = scaler_inputs.transform(u_id)
        return x_est, u_id


def scaler_inverse(x_est, path_scaler):
    scaler_states = load(open(path_scaler + "scaler_states.pkl", "rb"))
    x_est = scaler_states.inverse_transform(x_est)

    return x_est


def scale_single_state(value, state, path_scaler):

    if state.split("_")[0] == "input":
        pos = np.where(inputs == state)[0][0]
        u_id = np.zeros(len(inputs)).reshape(1, -1).astype(float)
        u_id[0, pos] = value

        scaler_inputs = load(open(path_scaler + "scaler_inputs.pkl", "rb"))
        u_id = scaler_inputs.transform(u_id)
        return u_id[0, pos]

    else:
        pos = np.where(states_name == state)[0][0]
        x_est = np.zeros(len(dict_states)).reshape(1, -1).astype(float)
        x_est[0, pos] = value

        scaler_states = load(open(path_scaler + "scaler_states.pkl", "rb"))
        x_est = scaler_states.transform(x_est)
        return x_est[0, pos]


def scale_inverse_Q1(value_array, path_scaler):
    scaler_states = load(open(path_scaler + "scaler_states.pkl", "rb"))
    scale = scaler_states.scale_[0]
    center = scaler_states.center_[0]

    value_array *= scale
    value_array += center

    return value_array
