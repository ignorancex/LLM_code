import glob
import os
import shutil

import autokeras as ak
import numpy as np
import tensorflow as tf

from callbacks import create_callbacks
from layers import MSEWeighted
from networks import NAS_automodel as model


class Trainer:
    def __init__(
        self,
        save_model_path,
        auto_model_params,
        update_params,
        nas_params,
        network_name,
        fit_params,
        options=None,
    ):

        self.network_name = network_name
        self.save_model_path = save_model_path

        self.auto_model_params = auto_model_params
        self.nas_params = nas_params
        self.update_params = update_params
        self.options = options

        self.loss = MSEWeighted(loss_affine=self.options["loss_affine"])
        self.compile_params = dict(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=self.loss,
            run_eagerly=False,
        )
        self.fit_params = fit_params

    def train(self, x_train, y_train, x_val, y_val):
        """Setup NAS graph and train NAS"""

        assert self.network_name in [
            "prosub",
            "sardunet-nas",
        ], "choose in {prosub|sardunet-nas}"
        graph_inputs_outputs = model(
            loss=self.loss,
            **self.nas_params,
        )

        auto_model = ak.AutoModel(**graph_inputs_outputs, **self.auto_model_params)
        callbacks = create_callbacks(**self.update_params)

        _ = auto_model.fit(
            **self.fit_params,
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
        )

        print("End of main NAS training")

    def clean_up_trials(self, save_model_path):
        trials = glob.glob(save_model_path + "/trial_*")
        for trial_dir in trials:
            print("Removing {}".format(trial_dir))
            shutil.rmtree(trial_dir)

    def load_predictor(self, postfix="last_model", model=None, model_print=False):
        # https://stackoverflow.com/questions/52800025/keras-give-input-to-intermediate-layer-and-get-final-output
        if model is None:
            print("Total model loaded from',save_model_path")
            if model_print:
                print(model.summary())
            save_model_path = os.path.join(self.save_model_path, postfix)
            model = tf.keras.models.load_model(
                save_model_path, custom_objects=ak.CUSTOM_OBJECTS, compile=False
            )

        m = model.get_layer("downsampling_mult_layer").get_weights()[0]
        sigma_bar = model.get_layer("downsampling_mult_layer").get_weights()[2]

        # Identify input index for predictor/reconstruction network
        S_end_layer = model.get_layer("DownsamplingOp")
        for idx, layer in enumerate(model.layers):
            if layer.name == S_end_layer.name:
                idx_in = idx + 1  # next one goes to the predictor input
        input_shape = model.layers[0].get_input_shape_at(0)

        # Build predictor/reconstruction network
        new_input = tf.keras.Input(shape=input_shape[1:])
        x = new_input
        for layer in model.layers[idx_in:]:
            x = layer(x)
        P_model = tf.keras.Model(inputs=new_input, outputs=x)
        if model_print:
            print("Predictor model", P_model.summary())

        return P_model, m, sigma_bar

    def evaluate(
        self,
        x_test,
        y_test,
        model=None,
        save_prediction=False,
        save_result=False,
        save_result_common=False,
    ):

        P_model, m, sigma_final = self.load_predictor(postfix="last_model", model=model)

        # Could also load sigma_final, m from saved dir (previous version)
        print("Number of nonzero-measurements (check):", np.sum(m != 0))
        x_test = sigma_final * m * x_test

        # prediction on test set
        y_pred = P_model.predict(
            x_test, batch_size=self.options["batch_size"], verbose=2
        )
        loss = np.mean(self.loss(y_test, y_pred))
        if save_prediction:
            np.save(os.path.join(self.save_model_path, "test_pred.npy"), y_pred)
        if save_result:
            np.savetxt(os.path.join(self.save_model_path, "test_result.txt"), [loss])
        if save_result_common:
            os.makedirs(
                os.path.join(self.options["out_base"], "results"), exist_ok=True
            )
            save_file = os.path.join(
                self.options["out_base"], "results", self.options["proj_name"] + ".npy"
            )
            if os.path.exists(save_file):
                print("overwriting saved result", save_file)
            print("Saving test result:", save_file)
            np.save(save_file, loss)

        return loss
