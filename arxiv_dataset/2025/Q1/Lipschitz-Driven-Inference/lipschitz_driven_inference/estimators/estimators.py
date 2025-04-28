from collections import namedtuple
from abc import abstractmethod

Dataset = namedtuple("Dataset", ["S", "X", "y"])
ConfidenceInterval = namedtuple("ConfidenceInterval", ["lower", "upper"])


class Estimator:
    def __init__(self, training_data: Dataset, test_data: Dataset):
        self.training_data = training_data
        self.test_data = test_data

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the estimator.
        """
        raise NotImplementedError

    @property
    def M(self) -> int:
        return self.test_data.S.shape[0]

    @property
    def N(self) -> int:
        return self.training_data.S.shape[0]

    @property
    def P(self) -> int:
        return self.training_data.X.shape[1]

    @abstractmethod
    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        raise NotImplementedError

    @abstractmethod
    def confidence_interval(self, dim: int, alpha: float = 0.05) -> ConfidenceInterval:
        """
        Compute the confidence interval for the estimator.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sd_estimate(self, dim: int) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        raise NotImplementedError
