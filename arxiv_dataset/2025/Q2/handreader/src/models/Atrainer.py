import abc


class AbstractTrainer(abc.ABC):
    """Abstract class for trainer."""

    @abc.abstractmethod
    def train_step(self, batch):
        """Defines a single training step.

        This method must be implemented by any subclass.
        """

    @abc.abstractmethod
    def validate_step(self, batch):
        """Defines a single validation step.

        This method must be implemented by any subclass.
        """

    @abc.abstractmethod
    def fit(self, train_loader, val_loader=None, epochs=1):
        """Runs the training and validation loops for a given number of epochs.

        This method must be implemented by any subclass.
        """
