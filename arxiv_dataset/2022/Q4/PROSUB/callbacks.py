import os

import numpy as np
import tensorflow as tf


def create_callbacks(
    network_name,
    tensorboard=False,
    **update_parameters,
):

    CSVlogger_path = os.path.join(
        update_parameters["save_model_path"], "val_result.csv"
    )
    # if os.path.isfile(self.CSVlogger_path): os.remove(self.CSVlogger_path)
    history = tf.keras.callbacks.History()
    csv_logger = tf.keras.callbacks.CSVLogger(
        CSVlogger_path, separator=",", append=True
    )
    custom_early_stopping = StopperAfterEpochs(
        start_epoch=update_parameters["epochs_decay"] * 2,
        patience=10,
        restore_best_weights=True,
    )
    callbacks = [history, csv_logger, custom_early_stopping]
    if tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                update_parameters["save_model_path"] + "/logs"
            )
        )

    if network_name in ["prosub", "prosub-nonas"]:
        callbacks.append(PROSUBCallback(**update_parameters))
        # add clear history later?
    elif network_name in ["sardunet-nas", "sardunet-nonas"]:
        callbacks.append(NASStepMSaveModel(**update_parameters))

    return callbacks


class StopperAfterEpochs(tf.keras.callbacks.EarlyStopping):
    """Early stopping starting after start_epoch,
    # From: https://stackoverflow.com/questions/46287403/is-there-a-way-to-implement-early-stopping-in-keras-only-after-the-first-say-1
    """

    def __init__(self, start_epoch, **kwargs):  # add argument for starting epoch
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class NASStepMSaveModel(tf.keras.callbacks.Callback):
    def __init__(self, T_values, M_values, save_model_path, **kwargs):
        super().__init__()
        assert len(T_values) == len(M_values)
        assert all(np.diff(T_values) > 0), "T_values[0] < T_values[1]..."
        assert all(np.diff(M_values) < 0), "M_values[0] > M_values[1]..."
        self.T_values = T_values
        self.M_values = M_values
        self.save_model_path = save_model_path

    def assign(self, **kwargs):
        self.model.get_layer("downsampling_mult_layer").assign(**kwargs)

    def get(self, *args):
        return self.model.get_layer("downsampling_mult_layer").get(*args)

    def on_train_begin(self, logs=None):
        self.t, M = self.get("t", "M")  # Init to 1 in layer
        # Initialized to 1, so need to set for first stage

        # Set M, note 1 < self.t < self.T == self.T_values[-1]
        if self.t < self.T_values[0]:
            # Initial condition for M
            M = self.M_values[0]
            self.assign(M_target=M, M_target_best_val=np.inf)
        else:
            if self.t in self.T_values:
                index = self.T_values.index(self.t)
                M = self.M_values[index + 1]
                self.assign(M_target=M, M_target_best_val=np.inf)
        self.assign(M=M)
        self.M = M
        print("M is", M, "in NAS step", self.t)
        print("M_target is ", self.get("M_target")[0], "in NAS step ", self.t)
        (m,) = self.get("m")
        print("m has", np.sum(np.sum(m[0] == 1)), "ones", np.sum(m[0] == 0), "zeros")

    def on_train_end(self, logs=None):
        # self.model.save(self.save_model_path+'/last_model',overwrite=True)
        self.assign(t=self.t + 1)

        self.on_train_end_save_vars()
        self.on_train_end_save_M_target_best_val(logs=logs)
        # Updates next step t+1

    def on_train_end_save_vars(self):
        if self.t + 1 in self.T_values:
            save_dir = os.path.join(self.save_model_path, "m" + str(self.M))
            os.makedirs(save_dir, exist_ok=True)
            m, sigma_bar = self.get("m", "sigma_bar")
            np.save(
                save_dir + "/vars.npy",
                {"M": self.M, "m": m, "sigma_bar": sigma_bar, "t": self.t},
            )
            print("saving vars in", save_dir + "/vars.npy")

    def on_train_end_save_M_target_best_val(self, logs):
        M_target, M_target_best_val = self.get("M_target", "M_target_best_val")
        if M_target != self.M_values[0]:
            if (
                logs["val_loss"] < M_target_best_val
            ):  # don't care about saving best model for M=N
                print(
                    "Current val_loss",
                    logs["val_loss"],
                    "is lower than",
                    M_target_best_val,
                    "for M=",
                    str(M_target),
                    "stage",
                )
                self.assign(M_target_best_val=logs["val_loss"])

                save_dir = os.path.join(self.save_model_path, "m" + str(self.M))
                os.makedirs(save_dir, exist_ok=True)
                print("Saving model in", save_dir)

                self.model.save(save_dir, overwrite=True)
            else:
                print(
                    "Current val_loss",
                    logs["val_loss"],
                    "is higher than",
                    M_target_best_val,
                    "for M=",
                    str(M_target),
                    "stage",
                )


class PROSUBCallback(NASStepMSaveModel):
    """alpha: scoring weights, total: sigma_bar = alpha * sigma+(1-alpha) * sigma_bar"""

    def __init__(
        self,
        T_values,
        M_values,
        save_model_path,
        measurements,
        epochs,
        epochs_decay,
        **kwargs,
    ):
        super().__init__(
            T_values=T_values, M_values=M_values, save_model_path=save_model_path
        )

        self.measurements = measurements
        self.D = np.zeros(
            T_values[-1] + 1, dtype=int
        )  # #measurements decay in NAS step t
        for i in range(len(self.M_values) - 1):
            t_start, t_end = (
                self.T_values[i],
                T_values[i + 1],
            )  # decay t=t_start...t_end-1
            nonsub_measurements = self.M_values[i] - M_values[i + 1]
            k_quo = nonsub_measurements // (t_end - t_start)
            k_remainder = nonsub_measurements % (t_end - t_start)
            self.D[t_start:t_end] = k_quo
            self.D[t_start] += k_remainder

        self.epochs_decay = epochs_decay
        assert epochs_decay <= epochs
        if epochs_decay <= 0:
            self.alpha_m = 1.0 / epochs  # initial value
        else:
            self.alpha_m = 1.0 / epochs_decay  # initial value
        print("set alpha_m set to", self.alpha_m)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        try:
            self.alpha_t = (self.T_values[-1] - self.t) / (self.T_values[-1] - 1)
            self.assign(alpha_t=self.alpha_t)
            self.set_m_decay()
        except:
            print("m_decay not set")

    def on_epoch_begin(self, epoch, logs=None):
        # super().on_epoch_begin(epoch=epoch, logs=logs)
        try:
            (m,) = self.get("m")
            if epoch >= self.epochs_decay:
                m = m - self.alpha_m * self.m_decay
            m = np.maximum(0, m)
            self.assign(m=m)
        except:
            print("Updating m in downsampling_mult_layer failed")

    def on_train_end(self, logs=None):
        # Update sigma_bar first before (potentially) saving model
        sigma, sigma_bar, alpha_t = self.get("sigma", "sigma_bar", "alpha_t")
        print("isnan in sigma:", np.count_nonzero(np.isnan(sigma)))
        sigma_bar_next = alpha_t * sigma + (1 - alpha_t) * sigma_bar
        self.assign(sigma_bar=sigma_bar_next)
        super().on_train_end(logs=logs)

    def set_m_decay(self):
        self.m_decay = np.zeros((1, self.measurements), dtype="int32")
        # Decay self.D[t] measurements in NAS step t
        D_t = self.D[self.t]  # D_{t} in paper
        if D_t > 0:
            # Find indices smallest sigma_bar where m!=0
            m, sigma_bar = self.get("m", "sigma_bar")
            assert np.sum((0 < m) & (m < 1)) == 0, "m has values between 0,1"
            assert np.sum(sigma_bar <= 0) == 0, "sigma_bar has values >= 0"

            m[np.where(m == 0)] = np.inf
            D = np.argsort(m * sigma_bar)[:, :D_t]  # D in paper
            self.m_decay[0][D] = 1
        print("Decay:", np.sum(self.m_decay), "measurements in NAS step", self.t)
