import torch
import torch.nn as nn


class ModelwoLLM(nn.Module):
    def __init__(self, backbone, task_head, token_mode="all"):
        super().__init__()
        self.backbone = backbone
        self.task_head = task_head
        self.token_mode = token_mode
    
    def forward(self, src_input, target=None, mode='train'):
        hidden_states = self.backbone(src_input)
        # all, cls, no_cls
        for i in range(len(hidden_states)):
            if self.token_mode == "cls":
                hidden_states[i] = hidden_states[i][:, 0:1]
            elif self.token_mode == "no_cls":
                hidden_states[i] = hidden_states[i][:, 1:]
        pred = self.task_head(hidden_states)
        return pred
