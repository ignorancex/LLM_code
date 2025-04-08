"""
Visdom Logger
------------------
"""

import logging
import os
from argparse import Namespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.base import (LightningLoggerBase,
                                            rank_zero_experiment)
from pytorch_lightning.utilities.logger import ( _convert_params,
                                                _flatten_dict)
from pytorch_lightning.utilities.logger import \
    _sanitize_params as _utils_sanitize_params
from pytorch_lightning.utilities.rank_zero import (rank_zero_only,
                                                   rank_zero_warn)
from visdom import Visdom

log = logging.getLogger(__name__)


class VisdomLogger(LightningLoggerBase):
    r"""
    Log to visdom.

    Example:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import VisdomLogger

        logger = VisdomLogger("localhost", name="my_model")
        trainer = Trainer(logger=logger)

    Args:
        server: Server address.
        name: Experiment name. Defaults to ``'default'``. If it is the empty string then no per-experiment
            subdirectory is used.
        \**args: 
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).

    """
    NAME_HPARAMS_FILE = "hparams.yaml"
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        server: str = None,
        name: Optional[str] = None,
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        **kwargs,
    ):
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._server = server or "localhost"
        self._name = name or f"UNAMED_{np.random.randint(9999999)}"
        self._version = version
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        
        self._experiment = None
        self.hparams = {}
        self._kwargs = kwargs

    @property
    def env(self):
        return f"{self.name}_{self.version}"

    @property
    @rank_zero_experiment
    def experiment(self) -> Visdom:
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        
        # TODO: Connect to visdom here
        self._experiment = Visdom(env=self.env,server='192.168.7.157', port=8097)
        # self._experiment = Visdom(env=self.env, server=self._server,base_url="/visdom", port=80, use_incoming_socket=False)

        return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record hyperparameters. TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the previously saved logs
        to display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
        """

        params = _convert_params(params)

        # store params to output
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        xlabel = 'step'
        if 'epoch' in metrics:
            step = metrics.pop('epoch')
            xlabel = 'epoch'
        
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                for line_name, line_value in v.items():
                    self.experiment.line([line_value], [step], win=k, opts={'ylabel': k,'xlabel':xlabel,'title': k,'showlegend':True}, name=line_name, update='append')
            else:
                try:
                    self.experiment.line([v], [step], win=k, opts={'ylabel': k,'xlabel':xlabel,'title': k}, update='append')
                # TODO: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array=None):
        if self._log_graph:
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                input_array = model._apply_batch_transfer_handler(input_array)
                model._running_torchscript = True
                self.experiment.add_graph(model, input_array)
                model._running_torchscript = False
            else:
                rank_zero_warn(
                    "Could not log computational graph since the"
                    " `model.example_input_array` attribute is not set"
                    " or `input_array` was not given",
                )

    def save(self) -> None:
        if self._experiment is not None:
            self._experiment.save([self.env])
        return super().save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Optional. Any code that needs to be run after training
        # finishes goes here
        if self._experiment is not None:
            self._experiment.use_socket = False
        self.save()
        
    @property
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):

        # vis = Visdom(server=self._server,base_url="/visdom", port=80, use_incoming_socket=False)
        vis = Visdom(server='192.168.7.157', port=8097)
        envs = [e for e in vis.get_env_list() if e.startswith(self.name)]
        vis.use_socket = False

        existing_versions = []
        for listing in envs:
            v = listing.split("_")[-1]
            existing_versions.append(int(v))
        if len(existing_versions) == 0:
            return 0
        
        return max(existing_versions) + 1
