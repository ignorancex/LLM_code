import torch
from avalanche.benchmarks import CLExperience
from avalanche.models import MultiTaskModule
from torch import nn


class CascadedScalingClassifier(MultiTaskModule):
    def __init__(
            self,
            in_features,
            future_classes=None,
            scale_each_class=True,
            scale=True,
            reset_scalers=False,
            always_combine=False,
            beta=10, gamma=1,
    ):
        super().__init__()

        self.in_features = in_features
        self.classifiers = torch.nn.ModuleDict()
        self.always_combine = always_combine

        self.past_scaling_heads = torch.nn.ModuleDict() if scale else None

        self.beta = beta
        self.gamma = gamma

        self.classes_seen_so_far = []

        self._stop = nn.Parameter(torch.randn(1))
        self.scale_each_class = scale_each_class
        self.reset_scalers = reset_scalers

        self.future_classes = future_classes
        self.future_layers = None
        if future_classes is not None and future_classes > 0:
            self.future_layers = nn.Linear(self.in_features, future_classes)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        device = self._adaptation_device
        curr_classes = experience.classes_in_this_experience

        if self.past_scaling_heads is not None and \
                len(self.past_scaling_heads) > 0 and self.reset_scalers:
            for p in self.past_scaling_heads.parameters():
                if hasattr(p, 'reset_parameters'):
                    p.reset_parameters()

        if (curr_classes not in self.classes_seen_so_far
                or len(self.classifiers) == 0):
            self.classes_seen_so_far.append(curr_classes)

            # for tid in set(task_labels):
            td = len(self.classifiers)
            tid = str(td)
            # head adaptation
            if tid not in self.classifiers:  # create new head
                past_classifiers = len(self.classifiers)

                new_head = nn.Linear(self.in_features, len(curr_classes)).to(
                    device)

                self.classifiers[tid] = new_head

                if past_classifiers > 0 and self.past_scaling_heads is not None:
                    scalers = nn.ModuleList([nn.Linear(self.in_features,
                                                       len(c) if self.scale_each_class else 1)
                                             for c in self.classes_seen_so_far])
                    self.past_scaling_heads[tid] = scalers

    def forward(self, x, task_labels=None):
        logits = [c(x) for c in self.classifiers.values()]

        if len(logits) > 1 and self.past_scaling_heads is not None:
            scalers = [[torch.sigmoid(self.gamma * s(x) + self.beta) for s in v]
                       for v in self.past_scaling_heads.values()]

            for i, (l, sig) in enumerate(zip(logits[1:], scalers)):
                for j, s in enumerate(sig):
                    logits[j] = logits[j] * s

        if self.always_combine:
            return torch.cat(logits, -1)

        future = None
        with torch.no_grad():
            if self.future_layers is not None and self.training:
                self.future_layers.reset_parameters()
                future = self.future_layers(x)

        return logits, future
