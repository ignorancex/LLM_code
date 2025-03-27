import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_score, \
    recall_score

import torch
import torch.nn as nn
import torch.optim as optim

from DeepLearning.training_loop import standard_training
from DeepLearning.self_attention_model import SelfAttentionModel
from LoadAndProcessData.prepare_processed_dataset_for_pytorch import create_df_for_pytorch_forecasting
from LoadAndProcessData.prepare_processed_dataset_for_multisequence_pytorch import \
    generate_multi_sequences_dataloaders_with_weighted_loss


def train_network(model, train_dataloader, validation_dataloader, device, learning_rate, repetitions,
                  weight_class=None, min_length=7, max_length=42, use_sample_weigth=True):
    if weight_class is None:
        loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    else:
        loss = nn.CrossEntropyLoss(reduction="none", weight=weight_class).to(device)
    # Define Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Define Scheduler
    precision = 1e-5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=10,
                                                     verbose=True, eps=precision)

    best_state = standard_training(model=model, cross_entropy_loss_for_class=loss, device=device,
                                   optimizer_classifier=optimizer, scheduler=scheduler, precision=precision,
                                   dataloaders={"train": train_dataloader,
                                                "val": validation_dataloader}, patience_increase=20,
                                   variable_length=True, min_length=min_length, max_length=max_length,
                                   use_sample_weight=use_sample_weigth)
    model.load_state_dict(best_state['state_dict'])
    torch.save(model.state_dict(), "../Weights/trained_network_%d.pt" % repetitions)


def test_network(test_dataloader, model, device, repetitions):
    weights = torch.load("../Weights/trained_network_%d.pt" % repetitions)
    model.load_state_dict(weights)
    model.eval()

    dict_ground_truth = {}
    dict_predictions = {}
    with torch.no_grad():
        for batch in test_dataloader:
            X, y, lengths = batch
            # print("Lengths: ", lengths)
            predictions = model(X.to(device), lengths)
            predictions = np.argmax(predictions.cpu().numpy(), axis=1).tolist()
            y = y.numpy().tolist()
            for i, length in enumerate(lengths):
                if length in dict_predictions:
                    dict_predictions[length].append(predictions[i])
                    dict_ground_truth[length].append(y[i])
                else:
                    dict_predictions[length] = [predictions[i]]
                    dict_ground_truth[length] = [y[i]]
    for key in dict_ground_truth:
        ground_truth_current_length, predictions_current_length = dict_ground_truth[key], dict_predictions[key]
        print("CURRENT LENGTH: ", key)
        print("Accuracy: ", accuracy_score(ground_truth_current_length, predictions_current_length))
        print("Balanced Accuracy: ", balanced_accuracy_score(ground_truth_current_length, predictions_current_length))
        print("F1 Score: ", f1_score(ground_truth_current_length, predictions_current_length))
        print("Confusion Matrix: \n", confusion_matrix(ground_truth_current_length, predictions_current_length))
        print("\n")
    return dict_ground_truth, dict_predictions


if __name__ == '__main__':
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0013059065594403963
    nhead = 4
    encoder_feedforward_dim = 32
    hidden_features_size_before_attention = 4
    dropout = 0.1
    number_of_encoder_modules = 3
    min_length = 7
    max_length = 42

    use_sample_weight = True
    path_results = "../Results/56_days_batch_64_var_len_self_attention_no_weight.txt"

    df = pd.read_csv("../Dataset/G-ICBT_AnonymizedInteractionDataset.csv")

    dataframe_in_timeseries = create_df_for_pytorch_forecasting(df=df, number_of_day_for_classification=max_length)

    print(dataframe_in_timeseries)
    list_user_id = dataframe_in_timeseries["user_id"].unique()
    user_ids_exploration = [45, 18, 299, 113, 42, 228, 142, 106, 65, 61, 264, 284, 83, 231, 133, 56, 118, 272, 92, 278,
                            290, 185, 285, 328, 338, 206, 111, 136, 157, 304, 204, 82, 123, 72, 26, 305, 164, 5, 273,
                            70, 137, 200, 242, 46, 20, 35, 171, 27, 47, 213, 119, 139, 16, 263, 9, 4, 301, 96, 76, 160,
                            236, 32, 218, 337, 319, 168, 309, 224, 80, 73, 85, 241, 266, 203, 140, 29, 220, 336, 19,
                            239, 268, 88, 64, 233, 227, 30, 101, 57, 167, 235, 252, 186, 54, 14, 128, 23, 182, 208, 317,
                            314]

    list_array_index_to_consider_test = list_user_id[~np.in1d(list_user_id, user_ids_exploration)]
    print(list_array_index_to_consider_test)
    print(np.shape(list_array_index_to_consider_test))

    dataframe_users_exploration = dataframe_in_timeseries.loc[
        dataframe_in_timeseries["user_id"].isin(user_ids_exploration)]
    dataframe_users_train_test = dataframe_in_timeseries.loc[
        dataframe_in_timeseries["user_id"].isin(list_array_index_to_consider_test)]

    number_examples_test_cross_validation = 11
    all_predictions, all_truth = [], []
    array_indexes = np.array(list(range(len(list_array_index_to_consider_test))))

    array_all_ground_truth, array_all_predictions = [], []
    for repetition in range(20):
        dict_ground_truth_current_repetition, dict_predictions_current_repetition = {}, {}
        for index in range(0, len(array_indexes), number_examples_test_cross_validation):
            list_indexes_test = array_indexes[index:index + number_examples_test_cross_validation]
            train_dataloader, validation_dataloader, test_dataloader, weight_class = \
                generate_multi_sequences_dataloaders_with_weighted_loss(df=dataframe_users_train_test,
                                                                        list_indexes_test=list_indexes_test,
                                                                        df_exploration=dataframe_users_exploration,
                                                                        min_length=min_length,
                                                                        batch_size=64)
            print(weight_class)
            model = SelfAttentionModel(n_features_input=2, nhead=nhead,
                                       hidden_features_size_before_attention=hidden_features_size_before_attention,
                                       encoder_feedforward_dim=encoder_feedforward_dim, dropout=dropout,
                                       num_layers=number_of_encoder_modules).to(device)

            train_network(model=model, train_dataloader=train_dataloader,
                          validation_dataloader=validation_dataloader,
                          device=device, learning_rate=learning_rate, repetitions=repetition,
                          weight_class=weight_class, use_sample_weigth=use_sample_weight)

            dict_ground_truth, dict_predictions = test_network(test_dataloader=test_dataloader, model=model,
                                                               device=device, repetitions=repetition)

            for key_length_sequence in sorted(dict_predictions.keys()):
                if key_length_sequence in dict_ground_truth_current_repetition:
                    dict_ground_truth_current_repetition[key_length_sequence].extend(
                        dict_ground_truth[key_length_sequence])
                    dict_predictions_current_repetition[key_length_sequence].extend(
                        dict_predictions[key_length_sequence])
                else:
                    dict_ground_truth_current_repetition[key_length_sequence] = dict_ground_truth[key_length_sequence]
                    dict_predictions_current_repetition[key_length_sequence] = dict_predictions[key_length_sequence]
        array_all_ground_truth.append(dict_ground_truth_current_repetition)
        array_all_predictions.append(dict_predictions_current_repetition)

    results = []
    for current_repetition in range(len(array_all_predictions)):
        for key_length_sequence in sorted(array_all_predictions[current_repetition].keys()):
            print("Length sequence: ", key_length_sequence)
            ground_truths, predictions = array_all_ground_truth[current_repetition][key_length_sequence], \
                                         array_all_predictions[current_repetition][key_length_sequence]
            print(np.shape(ground_truths))

            tn, fp, fn, tp = confusion_matrix(ground_truths, predictions).ravel()
            print("Accuracy: ", accuracy_score(ground_truths, predictions))
            print("Balanced Accuracy: ", balanced_accuracy_score(ground_truths, predictions))
            print("Sensitivity: ", recall_score(ground_truths, predictions))
            print("Specificity: ", tn / (tn + fp))
            print("Precision: ", precision_score(ground_truths, predictions))
            print("F1 Score: ", f1_score(ground_truths, predictions))
            print("Confusion Matrix: \n", confusion_matrix(ground_truths, predictions))
            print("\n\n")
            results.append(
                {"Learning Rate": learning_rate, "nhead": nhead, "encoder_feedforward_dim": encoder_feedforward_dim,
                 "hidden_features_size_before_attention": hidden_features_size_before_attention,
                 "Balanced Accuracy": balanced_accuracy_score(ground_truths, predictions),
                 "F1 Score": f1_score(ground_truths, predictions),
                 "Confusion Matrix": confusion_matrix(ground_truths, predictions),
                 "Sequence Length": key_length_sequence, "Current Repetition": current_repetition,
                 "ground_truths": ground_truths, "predictions": predictions})

    print(results)
    with open(path_results, "wb") as file:
        pickle.dump(results, file=file)
