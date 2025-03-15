from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier as kNN_Classifier
from sklearn.neural_network import MLPClassifier as MLP_Classifier
from sklearn.svm import SVC as SVM_Classifier
import numpy as np
import torch

from ExpToolKit.models import BaseAE

from .const import CLASSIFIER_CONST
from .utils import dimension_reduction


class Classifier(object):
    def __init__(
        self, classifier_type: str, model: BaseAE | None = None, **kwargs
    ) -> None:
        """
        Initialize a specific type of classifier based on the classifier_type argument and additional arguments.

        Parameters
        ----------
        classifier_type : str
            The type of classifier to be used.
            Available options are:
                CLASSIFIER_CONST.KNN_CLASSIFIER
                CLASSIFIER_CONST.MLP_CLASSIFIER
                CLASSIFIER_CONST.SVM_CLASSIFIER
        model : BaseAE | None
            The model to be used for feature extraction.
            If None, the raw data is used.
        **kwargs
            Additional keyword arguments for the classifier.
        """
        self.model = model

        if classifier_type == CLASSIFIER_CONST.KNN_CLASSIFIER:
            self.classifier = kNN_Classifier(**kwargs)
        elif classifier_type == CLASSIFIER_CONST.MLP_CLASSIFIER:
            self.classifier = MLP_Classifier(**kwargs)
        elif classifier_type == CLASSIFIER_CONST.SVM_CLASSIFIER:
            self.classifier = SVM_Classifier(**kwargs)
        else:
            raise ValueError("Unknown classifier type")

    def fit(self, dataloader: DataLoader) -> float:
        """
        Fit the classifier based on the dataloader and return the train accuracy.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader for the data to be classified.

        Returns
        -------
        train_accuracy : float
            The accuracy of the classifier on the training set.
        """
        x_train = []
        y_train = []
        for x, y in dataloader:
            x: torch.Tensor
            y: torch.Tensor
            x_train.append(dimension_reduction(model=self.model, x=x))
            y_train.append(y.detach().numpy())
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        self.classifier = self.classifier.fit(x_train, y_train)

        train_accuracy = self.classifier.score(x_train, y_train)

        return train_accuracy

    def predict(self, dataloader: DataLoader) -> float:
        """
        Predict the classifier based on the dataloader and return the test accuracy.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader for the data to be predicted.

        Returns
        -------
        test_accuracy : float
            The accuracy of the classifier on the test set.
        """
        x_test = []
        y_test = []
        for x, y in dataloader:
            x: torch.Tensor
            y: torch.Tensor
            x_test.append(dimension_reduction(model=self.model, x=x))
            y_test.append(y.detach().numpy())
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        test_accuracy = self.classifier.score(x_test, y_test)

        return test_accuracy

    def evaluate(self, dataloader: DataLoader) -> float:
        pass
