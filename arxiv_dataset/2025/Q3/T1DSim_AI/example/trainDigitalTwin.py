import numpy as np
import torch
import time
import pandas as pd

from t1dsim_ai.options import (
    input_ind,
)
from t1dsim_ai.individual_model import IndividualModel, SequenceSelection
from t1dsim_ai.utils.metrics import (
    get_TIR,
    get_TBR70,
    get_TAR180,
    get_glucose_variability,
)
from t1dsim_ai.utils.preprocess import scale_inverse_Q1


def trainModel(
    df_data_subj,
    personalization_path,
    hidden_compartments,
    lr,
    batch_size,
    n_epochs,
    overlap,
    seq_len,
    subj="DT1",
):

    initTime = time.time()

    np.random.seed(0)
    torch.manual_seed(0)

    if (
        len(df_data_subj[~df_data_subj.is_train]) == 0
        or len(df_data_subj[df_data_subj.is_train]) == 0
    ):
        return None
    if (
        df_data_subj.loc[~df_data_subj.is_train, "input_meal_carbs"].sum() < 1
        or df_data_subj.loc[df_data_subj.is_train, "input_meal_carbs"].sum() < 1
    ):
        return None

    print("TRAINING SUBJECT: ", subj)

    NNIndividual = IndividualModel(subj, df_data_subj, personalization_path)
    NNIndividual.setup_nn(hidden_compartments, lr, batch_size, n_epochs, overlap,seq_len+1)
    score = NNIndividual.fit(True)

    if score is np.nan:
        return None
    else:
        df_info = pd.DataFrame(index=[0])

        df_info["subjectID"] = subj
        # df_info[['TrtGroup','sex','age','weight','height','insulin_modality']] = df_data_subj[['TrtGroup','sex','age','weight','height','insulin_modality']].iloc[0]
        df_info["TIR"] = 100 * get_TIR(df_data_subj.output_cgm)
        df_info["TAR"] = 100 * get_TAR180(df_data_subj.output_cgm)
        df_info["TBR"] = 100 * get_TBR70(df_data_subj.output_cgm)
        df_info["GlucoseVariability"] = 100 * get_glucose_variability(
            df_data_subj.output_cgm
        )

        data = {
            "train": [
                NNIndividual.x_est_train,
                NNIndividual.u_pop_train,
                NNIndividual.y_id_train,
                NNIndividual.u_ind_train,
                NNIndividual.cgm_real_train,
            ],
            "test": [
                NNIndividual.x_est_test,
                NNIndividual.u_pop_test,
                NNIndividual.y_id_test,
                NNIndividual.u_ind_test,
                NNIndividual.cgm_real_test,
            ],
        }

        print("TRAINING SUBJECT: ", subj)
        for group in ["train", "test"]:
            batch = SequenceSelection(seq_len+1, "cpu", data[group])
            with torch.no_grad():
                (
                    batch_x0_hidden,
                    batch_u_pop,
                    batch_u_ind,
                    batch_y,
                    batch_x_original,
                ) = batch.get_all("all")

                df_info["n_seq_" + group] = batch_x0_hidden.shape[0]

                df_info["TIR_" + group] = 100 * np.mean(
                    np.apply_along_axis(
                        get_TIR,
                        0,
                        scale_inverse_Q1(
                            batch_x_original[1:, :, [0]], NNIndividual.popModelFolder
                        )
                        .detach()
                        .numpy(),
                    )
                )
                df_info["TAR_" + group] = 100 * np.mean(
                    np.apply_along_axis(
                        get_TAR180,
                        0,
                        scale_inverse_Q1(
                            batch_x_original[1:, :, [0]], NNIndividual.popModelFolder
                        )
                        .detach()
                        .numpy(),
                    )
                )
                df_info["TBR_" + group] = 100 * np.mean(
                    np.apply_along_axis(
                        get_TBR70,
                        0,
                        scale_inverse_Q1(
                            batch_x_original[1:, :, [0]], NNIndividual.popModelFolder
                        )
                        .detach()
                        .numpy(),
                    )
                )

                for model_bool in [False, True]:
                    model = "AIPop" if not model_bool else "AIDT"
                    batch_x_sim = NNIndividual.nn_solution(
                        batch_x0_hidden, batch_u_pop, batch_u_ind, model_bool
                    )

                    df_info["RMSE_" + model + "_" + group] = torch.mean(
                        torch.sqrt(
                            torch.nanmean(
                                (
                                    scale_inverse_Q1(
                                        batch_x_sim[1:, :, [0]],
                                        NNIndividual.popModelFolder,
                                    )
                                    - scale_inverse_Q1(
                                        batch_x_original[1:, :, [0]],
                                        NNIndividual.popModelFolder,
                                    )
                                )
                                ** 2,
                                dim=0,
                            )
                        )
                    ).item()
                    df_info["TIR_" + model + "_" + group] = 100 * np.mean(
                        np.apply_along_axis(
                            get_TIR,
                            0,
                            scale_inverse_Q1(
                                batch_x_sim[1:, :, [0]], NNIndividual.popModelFolder
                            )
                            .detach()
                            .numpy(),
                        )
                    )
                    df_info["TAR_" + model + "_" + group] = 100 * np.mean(
                        np.apply_along_axis(
                            get_TAR180,
                            0,
                            scale_inverse_Q1(
                                batch_x_sim[1:, :, [0]], NNIndividual.popModelFolder
                            )
                            .detach()
                            .numpy(),
                        )
                    )
                    df_info["TBR_" + model + "_" + group] = 100 * np.mean(
                        np.apply_along_axis(
                            get_TBR70,
                            0,
                            scale_inverse_Q1(
                                batch_x_sim[1:, :, [0]], NNIndividual.popModelFolder
                            )
                            .detach()
                            .numpy(),
                        )
                    )

        df_info["train_epochs"] = NNIndividual.curr_epoch
        print(df_info.iloc[0])
        df_info.T.to_csv(NNIndividual.pathModel + "/info.csv")

        print("Training time:", (time.time() - initTime) / 60)


if __name__ == "__main__":
    # Architecture individual models
    n_neurons = 128

    hidden_compartments = {
        "models": [5 + len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
    }

    lr = 10 ** (-4)
    batch_size = 32
    n_epochs = 150
    overlap = 0.9
    seq_len = 60

    df_data_subj = pd.read_csv("example_model/data_example.csv")
    trainModel(
        df_data_subj,
        "example_model/",
        hidden_compartments,
        lr,
        batch_size,
        n_epochs,
        overlap,
        seq_len,
        "DT_Example",
    )
