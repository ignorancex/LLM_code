import os
from typing import Optional

import torch
from avalanche.benchmarks import CLExperience
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.models import MultiTaskModule
from torch import cosine_similarity, nn


class CumulativeMultiHeadClassifier(MultiTaskModule):

    def __init__(
            self,
            in_features,
    ):
        super().__init__()
        self.classes_so_far = 0
        self.in_features = in_features
        self.classifiers = nn.ModuleDict()

    def adaptation(self, experience: CLExperience):

        curr_classes = experience.classes_in_this_experience
        task_labels = experience.task_labels
        # if isinstance(task_labels, ConstantSequence):
        #     # task label is unique. Don't check duplicates.
        #     task_labels = [task_labels[0]]

        tid = str(task_labels[0])

        if tid not in self.classifiers:  # create new head
            new_head = nn.Linear(
                self.in_features, len(curr_classes)
            )
            self.classifiers[tid] = new_head

            # super().adaptation(experience)
            self.classes_so_far += len(curr_classes)

    def forward(
            self, x: torch.Tensor, task_labels: torch.Tensor, mask=True,
    ) -> torch.Tensor:
        """compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, mask)
        else:
            unique_tasks = torch.unique(task_labels)

        out = torch.zeros(x.shape[0], self.max_class_label, device=x.device)
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(), mask)
            assert len(out_task.shape) == 2, (
                "multi-head assumes mini-batches of 2 dimensions "
                "<batch, classes>"
            )
            n_labels_head = out_task.shape[1]
            out[task_mask, :n_labels_head] = out_task
        return out

    def forward_single_task(self, x, task_label, mask=True):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """

        return self.classifiers[str(task_label)](x)

        # out = torch.cat([c(x) for c in self.classifiers.values()], -1)
        #
        # return out

        # if task_label == 0:
        #     task_label = str(task_label)
        # o = self.classifiers[str(task_label)](x)
        # return o

        # else:
        out = torch.cat([self.classifiers[str(t)](x)
                         for t in range(task_label + 1)], -1)

        diff = abs(self.classes_so_far - out.shape[-1])

        if diff != 0 and mask:
            # out = torch.zeros(len(o), self.classes_so_far, device=o.device)
            out = nn.functional.pad(out, (0, diff), value=0)
            # out[:, :o.shape[-1]] = o
            # out[:, o.shape[-1]:] = -torch.inf

            # return out

        return out


def calculate_distance(x, y,
                       distance: str = None,
                       sigma=2,
                       normalize=False):
    if distance is None:
        distance = 'euclidean'

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception(f'{x.shape} {y.shape}')

    if normalize:
        x = torch.nn.functional.normalize(x, 2, -1)
        y = torch.nn.functional.normalize(y, 2, -1)

    a = x.unsqueeze(1).expand(n, m, d)
    b = y.unsqueeze(0).expand(n, m, d)

    if distance == 'euclidean':
        similarity = torch.pow(a - b, 2).sum(2).sqrt()
    elif distance == 'rbf':
        similarity = torch.pow(a - b, 2).sum(2).sqrt()
        similarity = similarity / (2 * sigma ** 2)
        similarity = torch.exp(similarity)
    elif distance == 'cosine':
        similarity = 1 - cosine_similarity(a, b, -1)
    else:
        assert False

    return similarity


def calculate_similarity(x, y,
                         distance: str = None,
                         sigma=2,
                         normalize=False):
    if distance is None:
        distance = 'euclidean'

    di = calculate_distance(x, y, distance, sigma, normalize)

    if distance == 'cosine':
        return 1 - di
    else:
        return - di


def get_save_path(scenario_name: str,
                  plugin: str,
                  plugin_name: str,
                  model_name: str,
                  exp_n: int = None,
                  sit: bool = False):
    base_path = os.getcwd()
    if exp_n is None:
        return os.path.join(base_path,
                            model_name,
                            scenario_name if not sit else
                            f'sit_{scenario_name}',
                            plugin,
                            plugin_name)

    experiment_path = os.path.join(base_path,
                                   model_name,
                                   scenario_name if not sit else
                                   f'sit_{scenario_name}',
                                   plugin,
                                   plugin_name,
                                   f'exp_{exp_n}')
    return experiment_path


class TrainableParameters(Metric[int]):
    def __init__(self):
        self._compute_cost: Optional[int] = 0

    def update(self, model: nn.Module):
        if hasattr(model, 'count_parameters'):
            self._compute_cost = model.count_parameters()
        else:
            self._compute_cost = sum(
                p.numel() for p in model.parameters() if p.requires_grad)

    def result(self) -> Optional[int]:
        return self._compute_cost

    def reset(self):
        self._compute_cost = 0


class TrainableParametersPlugin(GenericPluginMetric):
    """
    At the end of each experience, this metric reports the
    MAC computed on a single pattern.
    This plugin metric only works at eval time.
    """
    def __init__(self, reset_at, emit_at, mode):
        self._trainable_parameters = TrainableParameters()

        super(TrainableParametersPlugin, self).__init__(
            self._trainable_parameters, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        self._trainable_parameters.update(strategy.model)


class ExperienceTrainableParameters(TrainableParametersPlugin):
    """
    At the end of each experience, this metric reports the
    MAC computed on a single pattern.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceMAC metric
        """
        super(ExperienceTrainableParameters, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "TrainableParameters_exp"


class StreamTrainableParameters(TrainableParametersPlugin):
    """
    At the end of each experience, this metric reports the
    MAC computed on a single pattern.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceMAC metric
        """
        super(StreamTrainableParameters, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "TrainableParameters_stream"