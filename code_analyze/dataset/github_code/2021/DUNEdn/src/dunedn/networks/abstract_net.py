from abc import ABC, abstractmethod
import logging
from time import time as tm
import torch
from .model_utils import MyDataParallel
from .utils import print_epoch_logs
from dunedn.training.callbacks import History, Callback, CallbackList
from dunedn.training.losses import Loss
from dunedn.training.metrics import MetricsList
from dunedn import PACKAGE

logger = logging.getLogger(PACKAGE + ".train")


class AbstractNet(torch.nn.Module, ABC):
    """Abstract network implementation.

    Allows network distribution on multiple devices.

    Example
    -------

    Instantiate an AbstractNet daughter class such as GcnnNet:

    >>> network = GcnnNet(setup)
    >>> print(type(network))
    <class 'dunedn.networks.gcnn_net.GcnnNet'>

    Distribute the network on multiple devices:

    >>> device_ids = [0, 1, 2, 3]
    >>> device_ids
    [0, 1, 2, 3]
    >>> dp = network.to_data_parallel(device_ids=device_ids)
    >>> print(type(dp))
    <class 'dunedn.networks.model_utils.MyDataParallel'>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__is_compiled = False
        self.history = History()

    @property
    def is_compiled(self):
        return self.__is_compiled

    @is_compiled.setter
    def is_compiled(self, value):
        self.__is_compiled = value

    def to_data_parallel(self, device_ids: list) -> MyDataParallel:
        """Returns the model wrapped by MyDataParallel class.

        Parameters
        ----------
        device_ids: list
            List of devices to place the model to. The first device in the list
            is the master device.

        Returns
        -------
        MyDataParallel
            The wrapped network for distributed training.

        """
        return MyDataParallel(self, device_ids=device_ids)

    def check_network_is_compiled(self):
        """Checks wether the object is compiled or not.

        Raises
        ------
        RuntimeError
            If the network is not compiled.
        """
        if self.__is_compiled == False:
            raise RuntimeError(
                "Model must be compiled before training/testing. "
                "Please call `model.compile()` method."
            )

    def compile(self, loss: Loss, optimizer: torch.optim.Optimizer, metrics: list[str]):
        """Compiles network.

        Adds loss function, optimizer and metrics functions as attributes.

        Parameters
        ----------
        loss: Loss
            The network loss function.
        optimizer: torch.optim.Optimizer
            The optimizer used to update network's parameters.
        metrics: list[str]
            List of metrics names.
        """
        self.loss_fn = loss
        # TODO: pass non defaults arguments to loss function
        self.optimizer = optimizer
        self.metrics_list = MetricsList(metrics)
        self.is_compiled = True

    def fit(
        self,
        train_generator: torch.utils.data.Dataset,
        epochs: int,
        val_generator: torch.utils.data.Dataset = None,
        dev: str = "cpu",
        callbacks: list[Callback] = None,
    ):
        """Main training function.

        Example
        -------

        Wcample with a GCNN network.

        Load a runcard.

        >>> import dunedn
        >>> runcard_path = Path("default.yaml")
        >>> setup = dunedn.utils.utils.load_runcard(runcard_path)

        Instantiate the network.

        >>> network = dunedn.networks.gcnn.train.load_and_compile_gcnn_network(
        ... "collection", setup["model"]["gcnn"])


        Load the training generator.

        >>> train_generator = dunedn.networks.gcnn.gcnn_dataloading.GcnnDataset(
        ... "train", dsetup=setup["dataset"])

        Train for one epoch.

        >>> history = network.fit(train_generator, epochs=1)

        Parameters
        ----------
        train_generator: torch.utils.data.Dataset
            The train dataset generator.
        epochs: int
            Number of epochs to train network on.
        val_generator: torch.utils.data.Dataset
            The validation dataset generator.
        dev: str
            The device hosting the computation. Defaults is "cpu".

        Returns
        -------
        history: History
            The history callback containing training details.
        """
        self.check_network_is_compiled()
        # train_sampler = DistributedSampler(dataset=train_data, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_generator,
            shuffle=True,
            # sampler=train_sampler,
            batch_size=train_generator.batch_size,
        )

        # create CallbackList object, add History object to it
        if callbacks is None:
            callbacks = []
        self.history = History()
        callbacks.append(self.history)
        self.callback_list = CallbackList(callbacks)

        # model on device
        self.to(dev)

        logger.info(f"Training for {epochs} epochs")

        self.callback_list.on_train_begin()

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch}/{epochs}")

            self.callback_list.on_epoch_begin()

            # training epoch
            start = tm()
            # epoch_logs = self.train_epoch(train_loader, dev)
            epoch_logs = {}
            epoch_time = tm() - start
            epoch_logs.update({"epoch": epoch, "epoch_time": epoch_time})

            # validation
            if val_generator is not None:
                _, logs = self.predict(val_generator, dev, verbose=0)
                val_logs = {f"val_{k}": v for k, v in logs.items()}
                epoch_logs.update(val_logs)

            print_epoch_logs(logger, self.metrics_names, epoch_logs)

            self.callback_list.on_epoch_end(epoch_logs)

        training_logs = None
        self.callback_list.on_train_end(training_logs)

        # model to cpu to save memory
        self.to("cpu")

        return self.history

    @abstractmethod
    def predict(self, generator: torch.utils.data.Dataset, device: str) -> torch.Tensor:
        """Network inference.

        Parameters
        ----------
        generator: torch.utils.data.Dataset
            The inference generator.
        device: str
            Device hosting computation.

        Returns
        -------
        y_pred: torch.Tensor
            Prediction tensor. Placed on "cpu" for GPU memory saving.
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        dev: str = "cpu",
    ) -> list[float]:
        """Trains the network for one epoch.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training dataloader.
        dev: str
            The device hosting the computation.
        callback_list:
            callbacks implementing  on_train_epoch_begin, on_train_epoch_end


        Returns
        -------
        epoch_history: dict
            Dictionary containing epoch history. Computed quantities
            at each optimization iteration, with their uncertainties. Keys:

            - loss (list[Tuple(float, float)])
            - metrics (list[Tuple(float, float)])
        """
        pass
