import numpy as np
import pandas as pd
from datetime import time
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset


def scale_examples(examples, scaler):
    scaled_examples = []
    for example in examples:
        scaled_examples.append(scaler.transform(example))
    return np.array(scaled_examples)


def get_weight_classes_and_normalization_statistics(examples, labels):
    scaler = StandardScaler()
    examples_to_scales = []
    for example_sequence in examples:
        for sequence in example_sequence:
            examples_to_scales.append(sequence)
    scaler.fit(examples_to_scales)

    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1. / class_sample_count

    samples_weight = [weight[t] if t == 0 else weight[t] for t in labels]

    return scaler, np.array(samples_weight)


def generate_unscaled_examples_labels(df):
    examples, labels = [], []
    list_user_id = df["user_id"].unique()
    for user_index in list_user_id:
        user_logs = df.loc[df["user_id"] == user_index].copy()
        user_example = user_logs[["connected_today", "total_length_session"]].copy()
        user_label = user_logs["class_label"].iloc[0]
        examples.append(user_example)
        labels.append(user_label)

    return np.array(examples), np.array(labels)


def get_weight_class_for_loss(labels):
    ratio = np.mean(labels)
    weight = ratio / (1. - ratio)
    weights_class = torch.tensor([weight, 1.]).type(torch.float32)
    print(weights_class)
    return weights_class


def generate_dataloaders_exploration_with_weighted_loss(df, list_indexes_test, percentage_validation_of_train=0.1,
                                                        batch_size=32):
    examples, labels = generate_unscaled_examples_labels(df)
    list_indexes = np.array(list(range(len(examples))))
    train_and_validation_indexes = list_indexes[~np.in1d(list_indexes, list_indexes_test)]
    list_index_labels_0 = np.where(labels[train_and_validation_indexes] == 0)[0]

    test_indexes = list_indexes_test
    randomize_index = np.random.choice(len(train_and_validation_indexes), len(train_and_validation_indexes),
                                       replace=False)
    validation_indexes = train_and_validation_indexes[
        randomize_index[0:int(len(train_and_validation_indexes) * percentage_validation_of_train)]]
    train_indexes = train_and_validation_indexes[
        randomize_index[int(len(train_and_validation_indexes) * percentage_validation_of_train):]]
    print(np.any(np.in1d(list_index_labels_0, validation_indexes)))
    if np.any(np.in1d(list_index_labels_0, validation_indexes)) == False:
        print("ADDING NEW LABEL 0 FOR VALIDATION")
        index_to_add_to_validation = np.random.choice(list_index_labels_0)
        validation_indexes = np.append(validation_indexes, index_to_add_to_validation)
        train_indexes = np.delete(train_indexes, np.where(train_indexes == index_to_add_to_validation))

    print("Train Indexes: ", train_indexes)
    print("Validation Indexes: ", validation_indexes)
    print("Test Indexes: ", test_indexes)
    print("train_and_validation_indexes: ", train_and_validation_indexes)
    print("Number of labels: ", len(labels))
    print("Validation labels: ", labels[validation_indexes])
    print("Training labels: ", labels[train_indexes])

    examples_train, labels_train = examples[train_indexes], labels[train_indexes]
    examples_validation, labels_validation = examples[validation_indexes], labels[validation_indexes]
    examples_test, labels_test = examples[test_indexes], labels[test_indexes]
    scaler, _ = get_weight_classes_and_normalization_statistics(examples=examples_train, labels=labels_train)
    weight_class = get_weight_class_for_loss(labels=labels_train)
    examples_train = scale_examples(examples=examples_train, scaler=scaler)
    examples_validation = scale_examples(examples=examples_validation, scaler=scaler)
    examples_test = scale_examples(examples=examples_test, scaler=scaler)

    train = TensorDataset(torch.from_numpy(np.array(examples_train, dtype=np.float32)),
                          torch.from_numpy(np.array(labels_train, dtype=np.int64)))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True)
    print("Data: ")
    for i, (data, target) in enumerate(train_dataloader):
        print(target)

    validation = TensorDataset(torch.from_numpy(np.array(examples_validation, dtype=np.float32)),
                               torch.from_numpy(np.array(labels_validation, dtype=np.int64)))
    validation_dataloader = torch.utils.data.DataLoader(validation, batch_size=batch_size,
                                                        drop_last=False)

    test = TensorDataset(torch.from_numpy(np.array(examples_test, dtype=np.float32)),
                         torch.from_numpy(np.array(labels_test, dtype=np.int64)))
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, validation_dataloader, test_dataloader, weight_class


def generate_dataloaders_with_weighted_loss(df, list_indexes_test, df_exploration, percentage_validation_of_train=0.1,
                                            batch_size=32):
    examples, labels = generate_unscaled_examples_labels(df)
    examples_explorations, labels_explorations = generate_unscaled_examples_labels(df_exploration)
    list_indexes = np.array(list(range(len(examples))))
    train_and_validation_indexes = list_indexes[~np.in1d(list_indexes, list_indexes_test)]
    list_index_labels_0 = np.where(labels[train_and_validation_indexes] == 0)[0]

    test_indexes = list_indexes_test

    randomize_index = np.random.choice(len(train_and_validation_indexes), len(train_and_validation_indexes),
                                       replace=False)
    validation_indexes = train_and_validation_indexes[
        randomize_index[0:int(len(train_and_validation_indexes) * percentage_validation_of_train)]]
    train_indexes = train_and_validation_indexes[
        randomize_index[int(len(train_and_validation_indexes) * percentage_validation_of_train):]]
    print(np.any(np.in1d(list_index_labels_0, validation_indexes)))
    if np.any(np.in1d(list_index_labels_0, validation_indexes)) == False:
        index_to_add_to_validation = np.random.choice(list_index_labels_0)
        validation_indexes = np.append(validation_indexes, index_to_add_to_validation)
        train_indexes = np.delete(train_indexes, np.where(train_indexes == index_to_add_to_validation))

    examples_train, labels_train = examples[train_indexes], labels[train_indexes]
    examples_validation, labels_validation = examples[validation_indexes], labels[validation_indexes]
    examples_test, labels_test = examples[test_indexes], labels[test_indexes]

    print("Before examples train: ", np.shape(examples_train))
    print("Before examples validation: ", np.shape(examples_validation))
    randomize_index = np.random.choice(len(labels_explorations), len(labels_explorations),
                                       replace=False)
    examples_explorations_training = examples_explorations[randomize_index][
                                     int(len(examples_explorations) * percentage_validation_of_train):]
    labels_explorations_training = labels_explorations[randomize_index][
                                   int(len(labels_explorations) * percentage_validation_of_train):]
    examples_explorations_validation = examples_explorations[randomize_index][
                                       0:int(len(examples_explorations) * percentage_validation_of_train)]
    labels_explorations_validation = labels_explorations[randomize_index][
                                     0:int(len(labels_explorations) * percentage_validation_of_train)]

    examples_train = np.concatenate((examples_train, examples_explorations_training))
    labels_train = np.concatenate((labels_train, labels_explorations_training))
    examples_validation = np.concatenate((examples_validation, examples_explorations_validation))
    labels_validation = np.concatenate((labels_validation, labels_explorations_validation))
    print("After examples train: ", np.shape(examples_train))
    print("After examples validation: ", np.shape(examples_validation))
    scaler, _ = get_weight_classes_and_normalization_statistics(examples=examples_train, labels=labels_train)
    weights_class = get_weight_class_for_loss(labels_train)

    examples_train = scale_examples(examples=examples_train, scaler=scaler)
    examples_validation = scale_examples(examples=examples_validation, scaler=scaler)
    examples_test = scale_examples(examples=examples_test, scaler=scaler)

    train = TensorDataset(torch.from_numpy(np.array(examples_train, dtype=np.float32)),
                          torch.from_numpy(np.array(labels_train, dtype=np.int64)))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True)
    print("Data: ")
    for i, (data, target) in enumerate(train_dataloader):
        print(target)

    validation = TensorDataset(torch.from_numpy(np.array(examples_validation, dtype=np.float32)),
                               torch.from_numpy(np.array(labels_validation, dtype=np.int64)))
    validation_dataloader = torch.utils.data.DataLoader(validation, batch_size=batch_size,
                                                        drop_last=False)

    test = TensorDataset(torch.from_numpy(np.array(examples_test, dtype=np.float32)),
                         torch.from_numpy(np.array(labels_test, dtype=np.int64)))
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, validation_dataloader, test_dataloader, weights_class


def create_df_for_pytorch_forecasting(df, number_of_day_for_classification=7):
    list_user_id = df["Anonymized ID"].unique()
    print("List user id: ", len(list_user_id.tolist()))

    dataframe_in_timeseries = []
    index_user = 0
    for user_id in list_user_id:
        df_current_user = df.loc[df["Anonymized ID"] == user_id].copy()
        user_label = df_current_user["Label"].iloc[0]
        start_date = pd.to_datetime(df_current_user["sessionlogtimestampstart"].iloc[0])
        index_old_df = 0
        current_session = pd.to_datetime(df_current_user["sessionlogtimestampstart"].iloc[index_old_df])
        total_length_session_current_day_in_seconds = 0
        connected_today = 0
        number_of_sessions_current_day = 0
        time_of_connection_first_session = time(0).hour
        print("Time connection first session: ", time_of_connection_first_session)

        dict_dataset_user = {"user_id": [], "day_idx": [], "connected_today": [], "number_of_sessions_current_day": [],
                             "total_length_session": [], "time_of_connection_first_session": [], "class_label": []}
        go_with_while = True
        for day_number in range(number_of_day_for_classification):
            number_of_days_elapsed_since_start_session = (current_session - start_date).days
            first_loop = True
            while number_of_days_elapsed_since_start_session <= day_number and go_with_while:
                if first_loop:
                    time_of_connection_first_session = current_session.time().hour
                    first_loop = False
                number_of_sessions_current_day += 1
                connected_today = 1
                total_length_session_current_day_in_seconds += pd.to_timedelta(
                    df_current_user["session_duration"].iloc[index_old_df]).seconds

                index_old_df += 1
                if index_old_df >= len(df_current_user["sessionlogtimestampstart"]):
                    go_with_while = False
                    break
                current_session = pd.to_datetime(df_current_user["sessionlogtimestampstart"].iloc[index_old_df])
                number_of_days_elapsed_since_start_session = (current_session - start_date).days

            dict_dataset_user["user_id"].append(index_user)
            dict_dataset_user["day_idx"].append(day_number)
            dict_dataset_user["connected_today"].append(connected_today)
            dict_dataset_user["number_of_sessions_current_day"].append(number_of_sessions_current_day)
            dict_dataset_user["total_length_session"].append(total_length_session_current_day_in_seconds)
            dict_dataset_user["time_of_connection_first_session"].append(time_of_connection_first_session)
            dict_dataset_user["class_label"].append(user_label)

            connected_today = 0
            number_of_sessions_current_day = 0
            total_length_session_current_day_in_seconds = 0
            time_of_connection_first_session = time(0).hour
        print(dict_dataset_user)
        dataframe_in_timeseries.append(pd.DataFrame(dict_dataset_user))
        index_user += 1
    df_dataset_timeseries = pd.concat(dataframe_in_timeseries)
    return df_dataset_timeseries
