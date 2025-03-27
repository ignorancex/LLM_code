import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from DeepLearning.deep_learning_utils import pad_collate_fn_classification, VaryingLengthCustomDataset


def get_weight_class_for_loss(labels):
    ratio = np.mean(labels)
    weight = ratio / (1. - ratio)
    weights_class = torch.tensor([weight, 1.]).type(torch.float32)
    print(weights_class)
    return weights_class


def get_normalization_statistics(examples):
    scaler = StandardScaler()
    examples_to_scales = []
    for example_sequence in examples:
        for sequence in example_sequence:
            examples_to_scales.append(sequence)
    scaler.fit(examples_to_scales)
    return scaler


def generate_dataset_variable_length(user_examples, user_labels, scaler, min_length=7):
    examples_variable_length, labels_variable_length = [], []
    for user_example, user_label in zip(user_examples, user_labels):
        for sequence_length in range(min_length, len(user_example) + 1):
            sequence_example = user_example[0:sequence_length]
            scaled_sequence_example = scaler.transform(sequence_example)
            examples_variable_length.append(scaled_sequence_example)
            labels_variable_length.append(user_label)
    print("Number of examples: ", len(labels_variable_length))
    return np.array(examples_variable_length, dtype=object), np.array(labels_variable_length)


def generate_unscaled_max_length_examples_labels(df):
    examples, labels = [], []
    list_user_id = df["user_id"].unique()
    for user_index in list_user_id:
        user_logs = df.loc[df["user_id"] == user_index].copy()
        user_example = user_logs[["connected_today", "total_length_session"]].copy()
        user_label = user_logs["class_label"].iloc[0]
        examples.append(user_example)
        labels.append(user_label)

    return np.array(examples), np.array(labels)


def generate_multi_sequences_dataloaders_exploration_with_weighted_loss(df, list_indexes_test,
                                                                        percentage_validation_of_train=0.1,
                                                                        batch_size=32, min_length=7):
    examples, labels = generate_unscaled_max_length_examples_labels(df)
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

    if np.any(np.in1d(list_index_labels_0, validation_indexes)) == False:
        index_to_add_to_validation = np.random.choice(list_index_labels_0)
        validation_indexes = np.append(validation_indexes, index_to_add_to_validation)
        train_indexes = np.delete(train_indexes, np.where(train_indexes == index_to_add_to_validation))

    examples_train, labels_train = examples[train_indexes], labels[train_indexes]
    examples_validation, labels_validation = examples[validation_indexes], labels[validation_indexes]
    examples_test, labels_test = examples[test_indexes], labels[test_indexes]
    scaler = get_normalization_statistics(examples=examples_train)
    examples_train, labels_train = generate_dataset_variable_length(user_examples=examples_train, min_length=min_length,
                                                                    user_labels=labels_train, scaler=scaler)
    weight_class = get_weight_class_for_loss(labels=labels_train)
    print("WEIGHT CLASS: ", weight_class)
    examples_validation, labels_validation = generate_dataset_variable_length(user_examples=examples_validation,
                                                                              user_labels=labels_validation,
                                                                              scaler=scaler, min_length=min_length)
    examples_test, labels_test = generate_dataset_variable_length(user_examples=examples_test, user_labels=labels_test,
                                                                  scaler=scaler, min_length=min_length)

    train = VaryingLengthCustomDataset(examples_train, torch.from_numpy(labels_train))
    dataloader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=True,
                                                   collate_fn=pad_collate_fn_classification)

    validation = VaryingLengthCustomDataset(examples_validation, torch.from_numpy(labels_validation))
    dataloader_validation = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False,
                                                        drop_last=False, collate_fn=pad_collate_fn_classification)

    test = VaryingLengthCustomDataset(examples_test, torch.from_numpy(labels_test))
    dataloader_test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False,
                                                  collate_fn=pad_collate_fn_classification)

    return dataloader_train, dataloader_validation, dataloader_test, weight_class


def generate_multi_sequences_dataloaders_with_weighted_loss(df, list_indexes_test, df_exploration,
                                                            percentage_validation_of_train=0.1, batch_size=32,
                                                            min_length=7):
    examples, labels = generate_unscaled_max_length_examples_labels(df)
    examples_explorations, labels_explorations = generate_unscaled_max_length_examples_labels(df_exploration)
    list_indexes = np.array(list(range(len(examples))))
    train_and_validation_indexes = list_indexes[~np.in1d(list_indexes, list_indexes_test)]
    print(train_and_validation_indexes)
    list_index_labels_0 = np.where(labels[train_and_validation_indexes] == 0)[0]

    test_indexes = list_indexes_test

    randomize_index = np.random.choice(len(train_and_validation_indexes), len(train_and_validation_indexes),
                                       replace=False)
    validation_indexes = train_and_validation_indexes[
        randomize_index[0:int(len(train_and_validation_indexes) * percentage_validation_of_train)]]
    train_indexes = train_and_validation_indexes[
        randomize_index[int(len(train_and_validation_indexes) * percentage_validation_of_train):]]
    if np.any(np.in1d(list_index_labels_0, validation_indexes)) == False:
        index_to_add_to_validation = np.random.choice(list_index_labels_0)
        validation_indexes = np.append(validation_indexes, index_to_add_to_validation)
        train_indexes = np.delete(train_indexes, np.where(train_indexes == index_to_add_to_validation))

    examples_train, labels_train = examples[train_indexes], labels[train_indexes]
    examples_validation, labels_validation = examples[validation_indexes], labels[validation_indexes]
    examples_test, labels_test = examples[test_indexes], labels[test_indexes]
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

    scaler = get_normalization_statistics(examples=examples_train)
    examples_train, labels_train = generate_dataset_variable_length(user_examples=examples_train, min_length=min_length,
                                                                    user_labels=labels_train, scaler=scaler)
    weight_class = get_weight_class_for_loss(labels=labels_train)
    print("WEIGHT CLASS: ", weight_class)
    examples_validation, labels_validation = generate_dataset_variable_length(user_examples=examples_validation,
                                                                              user_labels=labels_validation,
                                                                              scaler=scaler, min_length=min_length)
    examples_test, labels_test = generate_dataset_variable_length(user_examples=examples_test, user_labels=labels_test,
                                                                  scaler=scaler, min_length=min_length)

    train = VaryingLengthCustomDataset(examples_train, torch.from_numpy(labels_train))
    dataloader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=True,
                                                   collate_fn=pad_collate_fn_classification)

    validation = VaryingLengthCustomDataset(examples_validation, torch.from_numpy(labels_validation))
    dataloader_validation = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False,
                                                        drop_last=False, collate_fn=pad_collate_fn_classification)

    test = VaryingLengthCustomDataset(examples_test, torch.from_numpy(labels_test))
    dataloader_test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False,
                                                  collate_fn=pad_collate_fn_classification)

    return dataloader_train, dataloader_validation, dataloader_test, weight_class
