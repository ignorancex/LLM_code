import ExpToolKit as etk
import os
import numpy as np
import pandas as pd
import pickle
import time
from ClassifierPack import Classifier, CLASSIFIER_CONST

seed = 2024
iteration = 10

for config_num in range(7):
    config_path = f"./model_config/config{config_num}.yaml"
    config = etk.load_config(config_path)
    model_name = etk.model_name(config)
    print(model_name)

    knn_train_acc, knn_test_acc = [np.zeros(iteration) for _ in range(2)]

    for i in range(iteration):
        print(f"\n{model_name}: Iter={i}")
        config["TRAIN"]["random_seed"] = seed + i

        train_settings = etk.create_train_setting(config)
        print(train_settings)

        model, train_loss_epoch, train_loss_batch, train_time_epoch, test_loss_epoch = (
            etk.train_and_test(**train_settings, is_print=True)
        )

        # classification
        train_loader, test_loader = etk.create_dataloader(config)

        knn_classifier = Classifier(
            CLASSIFIER_CONST.KNN_CLASSIFIER, model, n_neighbors=1, algorithm="brute"
        )
        knn_train_acc[i], knn_test_acc[i] = etk.evaluate_classifier(
            knn_classifier, train_loader, test_loader
        )

    # Classfication Results
    Acc_train_knn, AccStd_train_knn = np.mean(knn_train_acc), np.std(knn_train_acc)
    Acc_test_knn, AccStd_test_knn = np.mean(knn_test_acc), np.std(knn_test_acc)

    print(f"Acc_train : {Acc_train_knn:.3f} (std={AccStd_train_knn:.3f})")
    print(f"Acc_train : {Acc_test_knn:.3f} (std={AccStd_test_knn:.3f})")
