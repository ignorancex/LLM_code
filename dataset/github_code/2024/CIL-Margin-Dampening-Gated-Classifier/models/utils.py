import numpy as np
import torch
from avalanche.benchmarks import CLExperience
from avalanche.benchmarks.utils import ConstantSequence
from avalanche.models import MultiTaskModule
from torch import nn


class CustomMultiHeadClassifier(MultiTaskModule):
    def __init__(self, in_features, heads_generator, out_features=None,
                 p=None):

        super().__init__()

        self.heads_generator = heads_generator
        self.in_features = in_features
        self.starting_out_features = out_features
        self.classifiers = torch.nn.ModuleDict()

    def adaptation(self, experience: CLExperience):
        # super().adaptation(dataset)
        curr_classes = experience.classes_in_this_experience
        task = experience.task_labels

        if isinstance(task, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task = [task[0]]

        for tid in set(task):
            tid = str(tid)  # need str keys
            if tid not in self.classifiers:

                if self.starting_out_features is None:
                    out = max(curr_classes) + 1
                else:
                    out = self.starting_out_features

                new_head = self.heads_generator(self.in_features, out)
                self.classifiers[tid] = new_head

    def forward_single_task(self, x, task_label, **kwargs):
        return self.classifiers[str(task_label)](x, **kwargs)


class AvalanceCombinedModel(MultiTaskModule):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

        # self.dropout = lambda x: x
        # if p is not None:
        #     self.dropout = Dropout(p)

    def forward_single_task(self, x: torch.Tensor, task_label: int,
                            return_embeddings: bool = False,
                            t=None):

        out = self.feature_extractor(x, task_labels=task_label)
        out = torch.flatten(out, 1)

        # out = self.dropout(out)

        logits = self.classifier(out, task_labels=task_label)

        if return_embeddings:
            return out, logits

        return logits

    def forward_all_tasks(self, x: torch.Tensor,
                          return_embeddings: bool = False,
                          **kwargs):

        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x,
                                                    task_id,
                                                    return_embeddings,
                                                    **kwargs)
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor = None,
                return_embeddings: bool = False,
                **kwargs) \
            -> torch.Tensor:

        if task_labels is None:
            return self.forward_all_tasks(x,
                                          return_embeddings=return_embeddings,
                                          **kwargs)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, return_embeddings,
                                            **kwargs)

        unique_tasks = torch.unique(task_labels)
        if len(unique_tasks) == 1:
            return self.forward_single_task(x, unique_tasks.item(),
                                            return_embeddings, **kwargs)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(),
                                                return_embeddings, **kwargs)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out


class PytorchCombinedModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


class bn_track_stats:
    def __init__(self, module: nn.Module, condition=True):
        self.module = module
        self.enable = condition

    def __enter__(self):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self, type, value, traceback):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True


class SimplexClassifier(nn.Module):
    def __init__(
            self,
            in_features,
            projection_space=1000,
            out_features=None,
            *kwargs
    ):
        super().__init__()

        self.projector = nn.Linear(in_features, projection_space)
        self.classifier = torch.tensor(self.dsimplex(projection_space),
                                       dtype=torch.float)
        self.classifier = nn.Parameter(self.classifier, requires_grad=False)

    @staticmethod
    def dsimplex(num_classes=10):
        def simplex_coordinates2(m):
            x = np.zeros([m, m + 1])
            np.fill_diagonal(x, 1.0)

            a = (1.0 - np.sqrt(float(1 + m))) / float(m)

            x[:, m] = a

            c = np.zeros(m)
            for i in range(0, m):
                s = 0.0
                for j in range(0, m + 1):
                    s = s + x[i, j]
                c[i] = s / float(m + 1)

            for j in range(0, m + 1):
                for i in range(0, m):
                    x[i, j] = x[i, j] - c[i]

            x = x / np.linalg.norm(x, axis=0, keepdims=True)

            return x

        ds = simplex_coordinates2(num_classes)
        return ds

    def forward(self, x, **kwargs):
        return self.projector(x) @ self.classifier
