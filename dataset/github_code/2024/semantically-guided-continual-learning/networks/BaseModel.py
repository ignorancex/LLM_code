import torch
import torch.nn as nn


class BaseIncModel(nn.Module):
    def __init__(self, classes_names, clip_model):
        super().__init__()
        self.num_class_per_class = []
        self.task_cls = []
        self.task_offset = []
        self.param_type = clip_model.dtype
        self.num_classes = len(classes_names)

    @property
    def dtype(self):
        return self.param_type

    def _init_weights(self):
        pass

    def increment_process(self, nums):
        self.num_class_per_class.append(nums)
        self.task_cls = torch.tensor(self.num_class_per_class)
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_state_dict(self):
        return self.state_dict()

    def save_state_dict(self, save_path):
        torch.save(self.state_dict(), save_path)
