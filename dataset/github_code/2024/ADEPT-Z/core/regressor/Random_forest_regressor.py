import os
import sys

import numpy as np
import pandas as pd
import torch
from pyutils.plot import plt, set_ms
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

set_ms()
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from pyutils.config import configs

from core import builder
from core.regressor.regressor_base import RegressorBase


class RandomForestRegression(RegressorBase):
    # initialize an ExtraTreesRegressor
    def __init__(self, param_dict, feature_name_list, model, gene_file_path):
        super().__init__(model, gene_file_path)
        self.param_dict = param_dict
        self.feature_name_list = feature_name_list
        self.regression_model = ExtraTreesRegressor()

    # read features of all genes from csv file
    def read_features_from_csv(self, feature_path):
        self.feature_df = pd.read_csv(feature_path)
        num_params = self.feature_df["num_param"].values

        cr_device_counts = self.feature_df["cr_device_counts"].values

        num_dc2 = self.feature_df["2"].values
        num_dc3 = self.feature_df["3"].values
        num_dc4 = self.feature_df["4"].values
        num_dc6 = self.feature_df["6"].values
        num_dc8 = self.feature_df["8"].values

        expressivity_scores = self.feature_df["expressivity_score"].values

        robust_scores = self.feature_df["robust_score"].values

        uniformity_scores_js = self.feature_df["uniformity_score"].values

        predicted_ideal_test_acc = self.feature_df["predicted_ideal_acc"].values

        self.features_List = []
        for i in range(len(num_params)):
            # self.features_List.append(np.concatenate((np.array([num_params[i]]), np.array([num_dc2[i]]), np.array([num_dc3[i]]), np.array([num_dc4[i]]), \
            #                                     np.array([num_dc6[i]]), np.array([num_dc8[i]]), np.array([cr_device_counts[i]]), np.array([robust_scores[i]]),\
            #                                     np.array([expressivity_scores[i]]), np.array([uniformity_scores_js[i]])), axis=0))
            self.features_List.append(
                np.concatenate(
                    (
                        np.array([num_params[i]]),
                        np.array([num_dc2[i]]),
                        np.array([num_dc3[i]]),
                        np.array([num_dc4[i]]),
                        np.array([num_dc6[i]]),
                        np.array([num_dc8[i]]),
                        np.array([cr_device_counts[i]]),
                        np.array([robust_scores[i]]),
                        np.array([expressivity_scores[i]]),
                        np.array([uniformity_scores_js[i]]),
                        np.array([predicted_ideal_test_acc[i]]),
                    ),
                    axis=0,
                )
            )
        print("length of feature_list:", len(self.features_List))
        print("Number of features:", len(self.features_List[0]))
        self.features = np.array(self.features_List)

    # perform random forest regression and save the model
    def get_regression_result(self, model_save_path):
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.ideal_test_acc, test_size=self.param_dict["test_size"], random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.noisy_test_acc,
            test_size=self.param_dict["test_size"],
            random_state=42,
        )

        self.regression_model = ExtraTreesRegressor(
            n_estimators=self.param_dict["n_estimators"],
            criterion=self.param_dict["criterion"],
            max_depth=self.param_dict["max_depth"],
            min_samples_split=self.param_dict["min_samples_split"],
            min_samples_leaf=self.param_dict["min_samples_leaf"],
            min_weight_fraction_leaf=self.param_dict["min_weight_fraction_leaf"],
            max_features=self.param_dict["max_features"],
            max_leaf_nodes=self.param_dict["max_leaf_nodes"],
            min_impurity_decrease=self.param_dict["min_impurity_decrease"],
            bootstrap=self.param_dict["bootstrap"],
            oob_score=self.param_dict["oob_score"],
            n_jobs=self.param_dict["n_jobs"],
            random_state=self.param_dict["random_state"],
            verbose=self.param_dict["verbose"],
            warm_start=self.param_dict["warm_start"],
            ccp_alpha=self.param_dict["ccp_alpha"],
            max_samples=self.param_dict["max_samples"],
        )

        self.regression_model.fit(self.X_train, self.y_train)

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(self.regression_model, model_save_path)

    # get importance scores for all features, print them
    def get_importance_scores(self):
        self.feature_importances = self.regression_model.feature_importances_
        feature_importances = zip(self.feature_name_list, self.feature_importances)
        self.feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True
        )
        for feature, importance in self.feature_importances:
            print(f"feature: {feature}, importance: {importance}")

    # plot the regression result
    def plot_regression_result(self, figure_save_path):
        y_train_pred = self.regression_model.predict(self.X_train)

        y_test_pred = self.regression_model.predict(self.X_test)

        y_pred = np.concatenate([y_train_pred, y_test_pred])
        y = np.concatenate([self.y_train, self.y_test])

        corr_train, _ = spearmanr(y_train_pred, self.y_train)
        corr_test, _ = spearmanr(y_test_pred, self.y_test)
        corr, _ = spearmanr(y_pred, y)
        print("Spearman Coefficient for training set: ", corr_train)
        print("Spearman Coefficient for test set: ", corr_test)
        print("Spearman Coefficient for all: ", corr)

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
        plt.title("Training Set Actual vs. Predicted")
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
        plt.title("Test Set Actual vs. Predicted")
        plt.legend()

        plt.tight_layout()
        plt.savefig(figure_save_path)

    # save predicted ideal test acc to the feature dataframe for noisy test acc prediction
    def save_predicted_test_acc(self, df_save_path):
        predicted_ideal_acc = self.regression_model.predict(self.features)
        new_df = self.feature_df
        new_df["predicted_ideal_acc"] = predicted_ideal_acc
        new_df.to_csv(df_save_path, index=False)


def main():
    device = torch.device("cuda:0")
    param_dict_ideal = dict(
        test_size=0.1,
        n_estimators=150,
        criterion="squared_error",
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0,
        max_features=6,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=42,
        verbose=0,
        warm_start=False,
        ccp_alpha=0,
        max_samples=None,
    )

    param_dict_noisy = dict(
        test_size=0.1,
        n_estimators=150,
        criterion="squared_error",
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0,
        max_features=5,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=42,
        verbose=0,
        warm_start=False,
        ccp_alpha=0,
        max_samples=None,
    )

    # feature_name_list = ['num_param', '2', '3', '4', '6', '8', 'cr_device_counts', 'robust_score', 'expressivity_score', 'uniformity_score']
    feature_name_list = [
        "num_param",
        "2",
        "3",
        "4",
        "6",
        "8",
        "cr_device_counts",
        "robust_score",
        "expressivity_score",
        "uniformity_score",
        "predicted_ideal_acc",
    ]

    gene_file_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/training_results/training_results_30epochs_ideal_all.csv"

    dataset = "mnist"
    model = "cnn"
    config_file = f"configs/{dataset}/{model}/train_baseline.yml"
    configs.load(config_file, recursive=True)
    model = builder.make_model(device=device, model_cfg=configs.model, random_state=42)

    # rf_regressor = RandomForestRegression(param_dict=param_dict_ideal, feature_name_list=feature_name_list, model=model, gene_file_path=gene_file_path)
    rf_regressor = RandomForestRegression(
        param_dict=param_dict_noisy,
        feature_name_list=feature_name_list,
        model=model,
        gene_file_path=gene_file_path,
    )

    # features_save_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/training_results/gene_features_ideal_all.csv"
    # regression_model_save_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/random_forest/random_forest_extra_trees.joblib"
    # figure_save_path="/home/zjian124/Desktop/ADEPT_Zero/figures/RF_prediction_results_ideal.png"
    # df_save_path="/home/zjian124/Desktop/ADEPT_Zero/checkpoint/training_results/gene_features_noisy_all.csv"

    features_save_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/training_results/gene_features_noisy_all.csv"
    regression_model_save_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/random_forest/random_forest_extra_trees_noisy.joblib"
    figure_save_path = (
        "/home/zjian124/Desktop/ADEPT_Zero/figures/RF_prediction_results_noisy.png"
    )
    df_save_path = "/home/zjian124/Desktop/ADEPT_Zero/checkpoint/training_results/gene_features_noisy_all.csv"

    # rf_regressor.save_features_to_csv(output_file_path=features_save_path)

    rf_regressor.read_features_from_csv(feature_path=features_save_path)

    rf_regressor.get_regression_result(model_save_path=regression_model_save_path)

    rf_regressor.get_importance_scores()

    rf_regressor.plot_regression_result(figure_save_path=figure_save_path)

    # rf_regressor.save_predicted_test_acc(df_save_path=df_save_path)


if __name__ == "__main__":
    main()
