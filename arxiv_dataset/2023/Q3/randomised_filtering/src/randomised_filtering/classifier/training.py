import os
import numpy as np

from .generator import BalancedDataGen


def training_cv(data, model, nb_folds, batch_size, epochs, base_path_to_model=None):
    """Train and test model on given, separated datasets in cross validation manner.

    Results are reported and model weights are dumped if path is given.
    """

    fold_len = [int(len(d) / nb_folds) for d in data]

    # train 5 models for cross-validation
    for fold in range(nb_folds):

        # generate masks that cut out the data folds
        test_mask = []

        for i in range(len(data)):
            tmp_mask = np.zeros(len(data[i]), dtype=np.bool)
            tmp_mask[int(fold * fold_len[i]) : int((fold + 1) * fold_len[i])] = True
            test_mask += [tmp_mask]

        train_mask = [np.invert(m) for m in test_mask]

        path_to_model = None
        if base_path_to_model:
            base_path, ext = os.path.splitext(base_path_to_model)
            path_to_model = f"{base_path}_{str(fold)}{ext}"

        print(f"Working on model {str(fold)}...")

        train_model(
            data_train=[_data[_mask] for _data, _mask in zip(data, train_mask)],
            data_test=[_data[_mask] for _data, _mask in zip(data, test_mask)],
            model=model,
            batch_size=batch_size,
            epochs=epochs,
            path_to_model=path_to_model,
        )


def train_model(data_train, data_test, model, batch_size, epochs, path_to_model=None):
    """Train and test model on given, separated datasets and report results.

    Model weights are dumped if path is given.
    """

    assert len(data_train) == len(
        data_test
    ), "Training and testing data must have same number of classes."

    # make data generators
    traingen_pn = BalancedDataGen(
        data=data_train,
        weights=[1] * len(data_train),
        batch_size=batch_size,
    )
    testgen_pn = BalancedDataGen(
        data=data_test, weights=[1] * len(data_test), batch_size=batch_size
    )

    # fit and evaluate model
    model.fit(traingen_pn, epochs=epochs, verbose=1)
    model.evaluate(testgen_pn, verbose=1)

    # print results
    for i, _data in enumerate(data_test):
        correct_predictions = 1 - np.sum(np.round(model.predict(_data))) / len(_data)
        print(f"Class {i}")
        print(f" - Correct predictions: {correct_predictions}")
        print(f" - Nb. of streamlines: {len(_data)}")
        print()

    # store model weights if requested
    if path_to_model:
        model.save(path_to_model)
        print(f"Model saved to '{path_to_model}'.")
