import os
import sys

import numpy as np
import pandas as pd
import torch
from pyutils.general import logger as lg
from pyutils.plot import plt, set_ms
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

set_ms()
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from pyutils.config import configs

from core import builder
from core.regressor.regressor_base import RegressorBase


class MLPRegression(RegressorBase):
    # initialize an MLPRegressor
    # def __init__(self, param_dict, feature_name_list, model, gene_file_path):
    def __init__(self, param_dict, model, gene_file_path):
        super().__init__(model, gene_file_path)
        self.param_dict = param_dict
        # self.feature_name_list = feature_name_list
        self.regression_model = MLPRegressor()

    # read features of all genes from csv file
    def read_features_from_csv(self, feature_path):
        self.feature_df = pd.read_csv(feature_path)

        exclude_columns = ["gene"]

        feature_columns = [
            col for col in self.feature_df.columns if col not in exclude_columns
        ]

        self.features_List = []

        for index, row in self.feature_df.iterrows():
            features = {col: row[col] for col in feature_columns}
            self.features_List.append(features)

        self.features = np.array(
            [list(feature.values()) for feature in self.features_List]
        )
        print(self.features[151])

        self.feature_names = feature_columns

    # perform MLP regression and save the model
    def get_regression_result_ideal(self, model_save_path):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.ideal_test_acc,
            test_size=self.param_dict["test_size"],
            random_state=42,
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.regression_model = MLPRegressor(
            hidden_layer_sizes=self.param_dict["hidden_layer_sizes"],
            activation=self.param_dict["activation"],
            solver=self.param_dict["solver"],
            alpha=self.param_dict["alpha"],
            batch_size=self.param_dict["batch_size"],
            learning_rate=self.param_dict["learning_rate"],
            learning_rate_init=self.param_dict["learning_rate_init"],
            power_t=self.param_dict["power_t"],
            max_iter=self.param_dict["max_iter"],
            shuffle=self.param_dict["shuffle"],
            random_state=self.param_dict["random_state"],
            tol=self.param_dict["tol"],
            verbose=self.param_dict["verbose"],
            warm_start=self.param_dict["warm_start"],
            momentum=self.param_dict["momentum"],
            nesterovs_momentum=self.param_dict["nesterovs_momentum"],
            early_stopping=self.param_dict["early_stopping"],
            validation_fraction=self.param_dict["validation_fraction"],
            beta_1=self.param_dict["beta_1"],
            beta_2=self.param_dict["beta_2"],
            epsilon=self.param_dict["epsilon"],
            n_iter_no_change=self.param_dict["n_iter_no_change"],
            max_fun=self.param_dict["max_fun"],
        )

        self.regression_model.fit(self.X_train, self.y_train)

        # print(self.regression_model.hidden_layer_sizes)
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(
            {"model": self.regression_model, "scaler": self.scaler}, model_save_path
        )

    # perform MLP regression and save the model
    def get_regression_result_noisy(self, model_save_path):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.noisy_test_acc,
            test_size=self.param_dict["test_size"],
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.regression_model = MLPRegressor(
            hidden_layer_sizes=self.param_dict["hidden_layer_sizes"],
            activation=self.param_dict["activation"],
            solver=self.param_dict["solver"],
            alpha=self.param_dict["alpha"],
            batch_size=self.param_dict["batch_size"],
            learning_rate=self.param_dict["learning_rate"],
            learning_rate_init=self.param_dict["learning_rate_init"],
            power_t=self.param_dict["power_t"],
            max_iter=self.param_dict["max_iter"],
            shuffle=self.param_dict["shuffle"],
            random_state=self.param_dict["random_state"],
            tol=self.param_dict["tol"],
            verbose=self.param_dict["verbose"],
            warm_start=self.param_dict["warm_start"],
            momentum=self.param_dict["momentum"],
            nesterovs_momentum=self.param_dict["nesterovs_momentum"],
            early_stopping=self.param_dict["early_stopping"],
            validation_fraction=self.param_dict["validation_fraction"],
            beta_1=self.param_dict["beta_1"],
            beta_2=self.param_dict["beta_2"],
            epsilon=self.param_dict["epsilon"],
            n_iter_no_change=self.param_dict["n_iter_no_change"],
            max_fun=self.param_dict["max_fun"],
        )

        self.regression_model.fit(self.X_train, self.y_train)

        # print(self.regression_model.hidden_layer_sizes)
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(
            {"model": self.regression_model, "scaler": self.scaler}, model_save_path
        )

    # get parameters in the prediction model
    def get_model_param(self):
        print("activation:", self.regression_model.activation)
        print("n_layers:", self.regression_model.n_layers_)
        print("n_outputs:", self.regression_model.n_outputs_)
        for i, (coef, intercept) in enumerate(
            zip(self.regression_model.coefs_, self.regression_model.intercepts_)
        ):
            print(f"Layer {i + 1}")
            print(f"Weights:\n{coef}")
            print(f"Biases:\n{intercept}\n")

    # plot the regression result
    def plot_regression_result(self, figure_save_path, corr_set):
        # use model to predict test acc for training dataset
        y_train_pred = self.regression_model.predict(self.X_train)

        # x_data = self.X_train[0]
        # y_prediction = self.regression_model.predict([x_data])
        # print(x_data)
        # print(y_prediction)
        # use model to predict test acc for testing dataset
        y_test_pred = self.regression_model.predict(self.X_test)

        y_pred = np.concatenate([y_train_pred, y_test_pred])
        y = np.concatenate([self.y_train, self.y_test])

        corr_train, _ = spearmanr(y_train_pred, self.y_train)
        corr_test, _ = spearmanr(y_test_pred, self.y_test)
        corr, _ = spearmanr(y_pred, y)
        print("Spearman Coefficient for training set: ", corr_train)
        print("Spearman Coefficient for test set: ", corr_test)
        print("Spearman Coefficient for all: ", corr)

        corr_set.append([corr_train, corr_test])

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_train, y_train_pred, alpha=0.6, color="blue", label="Train")
        # plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
        plt.plot(
            [min(self.y_train), max(self.y_train)],
            [min(self.y_train), max(self.y_train)],
            "k--",
            lw=2,
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"Training Set Actual vs. Predicted\nSpearman Coefficient: {corr_train:.4f}"
        )
        plt.legend()

        # actual value and predicted_value
        plt.subplot(1, 2, 2)
        plt.scatter(self.y_test, y_test_pred, alpha=0.6, color="red", label="Test")
        # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.plot(
            [min(self.y_test), max(self.y_test)],
            [min(self.y_test), max(self.y_test)],
            "k--",
            lw=2,
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"Test Set Actual vs. Predicted\nSpearman Coefficient: {corr_test:.4f}"
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(figure_save_path)

    # save predicted ideal test acc to the feature dataframe for noisy test acc prediction
    def save_predicted_test_acc(self, model_save_path, df_save_path):
        # load model and scaler:
        model_and_scaler = joblib.load(model_save_path)
        model = model_and_scaler["model"]
        scaler = model_and_scaler["scaler"]
        # print("features:", self.features)
        # print("shape of features:", self.features.shape)
        inputs = scaler.transform(self.features)
        predictions = model.predict(inputs)
        predicted_ideal_acc = predictions * 5 + 95
        new_df = self.feature_df
        new_df["predicted_ideal_acc"] = predicted_ideal_acc
        new_df.to_csv(df_save_path, index=False)
        # print("predictions:", predictions)
        # print("length of predictions:", len(predictions))
        # print("1st feature:", self.features[0])

        # predicted_ideal_acc = (self.regression_model.predict(self.features) * 5) + 95
        # new_df = self.feature_df
        # new_df["predicted_ideal_acc"] = predicted_ideal_acc
        # new_df.to_csv(df_save_path, index=False)


def test_MLP_regressor_ideal():
    device = torch.device("cuda:0")

    hidden_layer_configurations = [(15, 14, 9)]

    corr_set = []
    for hidden_layers in hidden_layer_configurations:
        MLP_param_dict = dict(
            test_size=0.2,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=0.0005,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.02,
            power_t=0.5,  # Only used when solver='sgd'
            max_iter=400,
            shuffle=True,  # Only used when solver='sgd’'or 'adam'
            # random_state=41,
            random_state=0,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,  # Only used when solver='sgd'.
            nesterovs_momentum=True,  #  Only used when solver='sgd' and momentum > 0.
            early_stopping=False,  # Only effective when solver='sgd' or 'adam'
            validation_fraction=0.1,  # Only used if early_stopping is True
            beta_1=0.9,  # Only used when solver='adam'
            beta_2=0.999,  # Only used when solver='adam'
            epsilon=1e-8,  # Only used when solver='adam'
            n_iter_no_change=10,  # Only used when solver='sgd'or 'adam'
            max_fun=15000,  # Only used when solver='lbfgs'
        )

        # feature_name_list = ['num_param', '2', '3', '4', '6', '8', 'cr_device_counts', 'robust_score', 'expressivity_score', 'uniformity_score']
        # gene_file_path = "./checkpoint/training_results/training_results_8/training_results_30epochs_ideal_all.csv"
        gene_file_path = "./checkpoint/training_results/training_results_16/training_results_30epochs_ideal_all.csv"

        dataset = "mnist"
        model = "cnn"
        config_file = f"configs/{dataset}/{model}/train_baseline_16.yml"
        configs.load(config_file, recursive=True)
        model = builder.make_model(
            device=device, model_cfg=configs.model, random_state=42
        )

        mlp_regressor = MLPRegression(
            param_dict=MLP_param_dict, model=model, gene_file_path=gene_file_path
        )

        # features_save_path = "./checkpoint/training_results/training_results_8/gene_features_ideal_all.csv"
        features_save_path = "./checkpoint/training_results/training_results_16/gene_features_ideal_all.csv"

        regression_model_save_path = (
            "./checkpoint/random_forest/MLP_regression_model_16.joblib"
        )
        figure_save_path = "./figures/MLP_prediction_results_ideal_16.png"

        # mlp_regressor.save_features_to_csv(output_file_path=features_save_path)

        mlp_regressor.read_features_from_csv(feature_path=features_save_path)

        mlp_regressor.get_regression_result_ideal(
            model_save_path=regression_model_save_path
        )

        # # # mlp_regressor.get_model_param()

        mlp_regressor.plot_regression_result(
            figure_save_path=figure_save_path, corr_set=corr_set
        )

        mlp_regressor.save_predicted_test_acc(
            model_save_path="./checkpoint/random_forest/MLP_regression_model_16.joblib",
            df_save_path="./checkpoint/training_results/training_results_16/gene_features_noisy_new.csv",
        )


def test_MLP_regressor_noisy():
    device = torch.device("cuda:0")
    hidden_layer_configurations = [
        (i, j, k)
        for i in range(8, 14)  # from 8 to 13
        for j in range(8, 14)
        for k in range(8, 14)
    ]

    # hidden_layer_configurations = [
    #     (i, j)
    # for i in range(4, 17)  # from 4 to 16
    # for j in range(4, 17)
    # ]

    # hidden_layer_configurations = [
    #     (7,8,6,4),
    #     (5,4),
    #     (4,8,8)
    # ]

    hidden_layer_configurations = [(11, 7)]

    corr_set = []
    for hidden_layers in hidden_layer_configurations:
        MLP_param_dict = dict(
            test_size=0.2,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=0.001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.02,
            power_t=0.5,  # Only used when solver='sgd'
            max_iter=300,
            shuffle=True,  # Only used when solver='sgd’'or 'adam'
            # random_state=41,
            random_state=0,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,  # Only used when solver='sgd'.
            nesterovs_momentum=True,  #  Only used when solver='sgd' and momentum > 0.
            early_stopping=False,  # Only effective when solver='sgd' or 'adam'
            validation_fraction=0.1,  # Only used if early_stopping is True
            beta_1=0.9,  # Only used when solver='adam'
            beta_2=0.999,  # Only used when solver='adam'
            epsilon=1e-8,  # Only used when solver='adam'
            n_iter_no_change=10,  # Only used when solver='sgd'or 'adam'
            max_fun=15000,  # Only used when solver='lbfgs'
        )

        # feature_name_list = ['num_param', '2', '3', '4', '6', '8', 'cr_device_counts', 'robust_score', 'expressivity_score', 'uniformity_score']
        # gene_file_path = "./checkpoint/training_results/training_results_8/training_results_30epochs_ideal_all.csv"
        gene_file_path = "./checkpoint/training_results/training_results_16/training_results_30epochs_ideal_all.csv"

        dataset = "mnist"
        model = "cnn"
        config_file = f"configs/{dataset}/{model}/train_baseline_16.yml"
        configs.load(config_file, recursive=True)
        model = builder.make_model(
            device=device, model_cfg=configs.model, random_state=42
        )

        mlp_regressor = MLPRegression(
            param_dict=MLP_param_dict, model=model, gene_file_path=gene_file_path
        )

        features_save_path = "./checkpoint/training_results/training_results_16/gene_features_noisy_new.csv"
        regression_model_save_path = (
            "./checkpoint/random_forest/MLP_regression_model_noisy_16.joblib"
        )
        figure_save_path = "./figures/MLP_prediction_results_noisy_16.png"

        # checkpoint_path = "./checkpoint/mnist/cnn/train_16_MZI/SuperOCNN__acc-98.77_epoch-90.pt"
        # solution_path = "./configs/mnist/genes/MZI_solution_16.txt"
        # mlp_regressor.save_features_to_csv(output_file_path=features_save_path, checkpoint_path=checkpoint_path, solution_path=solution_path)

        mlp_regressor.read_features_from_csv(feature_path=features_save_path)

        mlp_regressor.get_regression_result_noisy(
            model_save_path=regression_model_save_path
        )

        # # # mlp_regressor.get_model_param()

        mlp_regressor.plot_regression_result(
            figure_save_path=figure_save_path, corr_set=corr_set
        )

        # mlp_regressor.save_predicted_test_acc(model_save_path="./checkpoint/random_forest/MLP_regression_model.joblib",
        #                                       df_save_path="./checkpoint/training_results/training_results_16/gene_features_noisy_new.csv")

    for i in range(len(corr_set)):
        print(hidden_layer_configurations[i])
        print(corr_set[i])


def test_saved_MLP_regressor_ideal():
    # input_features1 = [[-0.67726107 -1.43192754 -0.41771199 -0.39247908 -0.04403993 -0.99081307 0.55062131 -0.55396261  0.04468992  1.02995149]]
    # input_features2 = [[-0.58960844 -1.12681566 -0.18788896 -0.30566725  0.39397264 -0.68506510.36890769  0.31863906 -1.13547059  0.86907563]]
    features_all = np.array(
        [
            [
                1.11111111e-01,
                9.16666667e-02,
                1.26037538e-01,
                1.63736150e-01,
                4.02960468e-05,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.11210317,
                0.01153846,
                0.1000201,
                0.39441022,
                0.063749,
                0.16346154,
                0.15384615,
                0.07692308,
                0.11538462,
                0.11538462,
            ],
            [
                0.13392857,
                0.0202381,
                0.09479226,
                0.40853158,
                0.04555905,
                0.16964286,
                0.17142857,
                0.07142857,
                0.17857143,
                0.10714286,
            ],
        ]
    )

    model_file_path_ideal = "./checkpoint/random_forest/MLP_regression_model_16.joblib"
    model_file_path = "./checkpoint/random_forest/MLP_regression_model_noisy_16.joblib"

    prediction_model_and_scaler_ideal = joblib.load(model_file_path_ideal)
    prediction_model_ideal = prediction_model_and_scaler_ideal["model"]
    prediction_scaler_ideal = prediction_model_and_scaler_ideal["scaler"]

    prediction_model_and_scaler = joblib.load(model_file_path)
    prediction_model = prediction_model_and_scaler["model"]
    prediction_scaler = prediction_model_and_scaler["scaler"]

    accuracy_list = []

    for features in features_all:
        inputs_ideal = prediction_scaler_ideal.transform([features])

        lg.info(f"Features for ideal prediction: {inputs_ideal}")

        predicted_ideal_acc = float(
            round(prediction_model_ideal.predict(inputs_ideal)[0], 4)
        )

        lg.info(f"predicted ideal_acc: {predicted_ideal_acc}")

        predicted_ideal_acc_array = np.array([predicted_ideal_acc]) * 5 + 95

        lg.info(f"predicted ideal_acc_array: {predicted_ideal_acc_array}")

        lg.info("Prediction of ideal accuracy get.")

        features_noisy = np.concatenate((features, predicted_ideal_acc_array))

        lg.info(f"Features for noisy prediction: {features_noisy}")

        inputs_noisy = prediction_scaler.transform([features_noisy])

        lg.info(f"Input_Noisy: {inputs_noisy}")

        accuracy = float(round(prediction_model.predict(inputs_noisy)[0], 4)) * 10 + 95

        lg.info(f"Noisy Accuracy Prediction: {accuracy}")

        accuracy_list.append(accuracy)
        lg.info("Prediction of noisy accuracy get.")


if __name__ == "__main__":
    # test_MLP_regressor_ideal()
    test_MLP_regressor_noisy()
    # test_saved_MLP_regressor_ideal()
    # test_saved_MLP_regressor_noisy()
