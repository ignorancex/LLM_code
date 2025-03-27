import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_score, \
    recall_score

import torch
import torch.nn as nn
import torch.optim as optim

from DeepLearning.training_loop import standard_training
from DeepLearning.self_attention_model import SelfAttentionModel
from LoadAndProcessData.prepare_processed_dataset_for_pytorch import create_df_for_pytorch_forecasting
from LoadAndProcessData.prepare_processed_dataset_for_multisequence_pytorch import \
    generate_multi_sequences_dataloaders_exploration_with_weighted_loss


def train_network(model, train_dataloader, validation_dataloader, device, learning_rate, repetitions,
                  weight_class=None, min_length=7, max_length=42):
    if weight_class is None:
        loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    else:
        loss = nn.CrossEntropyLoss(reduction="mean", weight=weight_class).to(device)
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
                                   variable_length=True, min_length=min_length, max_length=max_length)
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
                # print("Dict Predictions: ", dict_predictions)
                # print("Dict Ground Truth: ", dict_ground_truth)
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

    min_length = 7
    max_length = 42

    df = pd.read_csv("../Dataset/G-ICBT_AnonymizedInteractionDataset.csv")
    dataframe_in_timeseries = create_df_for_pytorch_forecasting(df=df, number_of_day_for_classification=max_length)
    print(dataframe_in_timeseries)
    list_user_id = dataframe_in_timeseries["user_id"].unique()
    print("KEYS: ", dataframe_in_timeseries.keys())
    print("Average label dataset: ", dataframe_in_timeseries['class_label'].mean())
    user_ids_exploration = [45, 18, 299, 113, 42, 228, 142, 106, 65, 61, 264, 284, 83, 231, 133, 56, 118, 272, 92, 278,
                            290, 185, 285, 328, 338, 206, 111, 136, 157, 304, 204, 82, 123, 72, 26, 305, 164, 5, 273,
                            70, 137, 200, 242, 46, 20, 35, 171, 27, 47, 213, 119, 139, 16, 263, 9, 4, 301, 96, 76, 160,
                            236, 32, 218, 337, 319, 168, 309, 224, 80, 73, 85, 241, 266, 203, 140, 29, 220, 336, 19,
                            239, 268, 88, 64, 233, 227, 30, 101, 57, 167, 235, 252, 186, 54, 14, 128, 23, 182, 208, 317,
                            314]

    dataframe_in_timeseries = dataframe_in_timeseries.loc[dataframe_in_timeseries["user_id"].isin(user_ids_exploration)]
    list_user_id = dataframe_in_timeseries["user_id"].unique()
    print(list_user_id)
    number_examples_test_cross_validation = 10
    # Randomly draw from a log-uniform distribution 100 values between 10e-5 and 10e1 to be use as the learning rate
    results_random_search = []
    path_hyperparameter_search = "../Results/HyperparametersSearch/56_random_search_expanded_variable_length.txt"
    if os.path.isfile(path_hyperparameter_search):
        with open(path_hyperparameter_search, "rb") as file:  # Unpickling
            results_random_search = pickle.load(file)

    for combination_hyperparameter_try_number in range(100):
        learning_rate = loguniform.rvs(1e-5, 1e0, size=1)[0]
        nhead = np.random.choice([1, 2, 4, 8])
        embeded_dimensions_possible = []
        current_embeded_dimension = nhead
        while current_embeded_dimension <= 128:
            embeded_dimensions_possible.append(current_embeded_dimension)
            current_embeded_dimension *= 2
        print("Number of heads: ", nhead)
        print("Embeded dimensions possible: ", embeded_dimensions_possible)
        encoder_feedforward_dim = np.random.choice([1, 2, 4, 8, 16, 32, 64, 128])
        hidden_features_size_before_attention = np.random.choice(embeded_dimensions_possible)
        dropout = np.random.choice([0., .1, .2, .3, .4, .5])
        number_of_encoder_module = np.random.choice([1, 2, 3])
        print("Learning rate: ", learning_rate, " Number of heads: ", nhead, " Embedded dim: ", encoder_feedforward_dim,
              " Hidden feature size: ", hidden_features_size_before_attention, " Dropout: ", dropout,
              " Number of encoder module: ", number_of_encoder_module)
        array_indexes = np.array(list(range(len(user_ids_exploration))))
        dict_all_ground_truth, dict_all_predictions = {}, {}
        for repetition in range(5):
            for index in range(0, len(array_indexes), number_examples_test_cross_validation):
                list_indexes_test = array_indexes[index:index + number_examples_test_cross_validation]
                print(list_indexes_test)

                train_dataloader, validation_dataloader, test_dataloader, weight_class \
                    = generate_multi_sequences_dataloaders_exploration_with_weighted_loss(df=dataframe_in_timeseries,
                                                                                          list_indexes_test=
                                                                                          list_indexes_test,
                                                                                          batch_size=64)

                model = SelfAttentionModel(n_features_input=2, nhead=nhead,
                                           hidden_features_size_before_attention=hidden_features_size_before_attention,
                                           encoder_feedforward_dim=encoder_feedforward_dim, dropout=dropout,
                                           num_layers=number_of_encoder_module).to(device)

                train_network(model=model, train_dataloader=train_dataloader,
                              validation_dataloader=validation_dataloader,
                              device=device, learning_rate=learning_rate, repetitions=repetition,
                              weight_class=weight_class, min_length=min_length, max_length=max_length)

                dict_ground_truth, dict_predictions = test_network(test_dataloader=test_dataloader, model=model,
                                                                   device=device, repetitions=repetition)

                for key_length_sequence in sorted(dict_predictions.keys()):
                    if key_length_sequence in dict_all_ground_truth:
                        dict_all_ground_truth[key_length_sequence].extend(dict_ground_truth[key_length_sequence])
                        dict_all_predictions[key_length_sequence].extend(dict_predictions[key_length_sequence])
                    else:
                        dict_all_ground_truth[key_length_sequence] = dict_ground_truth[key_length_sequence]
                        dict_all_predictions[key_length_sequence] = dict_predictions[key_length_sequence]

        balanced_accuracies = {}
        f1_scores = {}
        for key_length_sequence in sorted(dict_all_predictions.keys()):
            print("Length sequence: ", key_length_sequence)
            ground_truths, predictions = dict_all_ground_truth[key_length_sequence], dict_all_predictions[
                key_length_sequence]
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
            f1_scores[key_length_sequence] = f1_score(ground_truths, predictions)
            balanced_accuracies[key_length_sequence] = balanced_accuracy_score(ground_truths, predictions)
        results_random_search.append(
            {"Learning Rate": learning_rate, "nhead": nhead, "encoder_feedforward_dim": encoder_feedforward_dim,
             "hidden_features_size_before_attention": hidden_features_size_before_attention, "dropout": dropout,
             "number_encoder_modules": number_of_encoder_module, "Balanced Accuracy": balanced_accuracies,
             "F1 Score": f1_scores})
        print("Results random search:")
        print(results_random_search)
        with open(path_hyperparameter_search, "wb") as file:
            pickle.dump(results_random_search, file=file)

        with open(path_hyperparameter_search, "rb") as file:  # Unpickling
            results_random_search = pickle.load(file)

    print("Results random search:")
    print(results_random_search)
    with open(path_hyperparameter_search, "wb") as file:
        pickle.dump(results_random_search, file=file)

    with open(path_hyperparameter_search, "rb") as file:  # Unpickling
        results_random_search = pickle.load(file)
    print(results_random_search)
