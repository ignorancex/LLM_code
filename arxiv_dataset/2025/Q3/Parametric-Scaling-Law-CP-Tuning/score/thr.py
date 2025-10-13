import torch

class THR(object):

    def __init__(self, score_type="softmax"):
        super().__init__()
        self.score_type = score_type
        if score_type == "Identity":
            self.transform = lambda x: x
        elif score_type == "softmax":
            self.transform = lambda x: torch.softmax(x, dim=- 1)
        elif score_type == "log_softmax":
            self.transform = lambda x: torch.log_softmax(x, dim=-1)
        elif score_type == "log":
            self.transform = lambda x: torch.log(x)
        else:
            raise NotImplementedError
    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        temp_values = self.transform(logits)
        if label is None:
            return self.__calculate_all_label(temp_values)
        else:
            return self.__calculate_single_label(temp_values, label)

    def __calculate_single_label(self, temp_values, label):
        return 1 - temp_values[torch.arange(temp_values.shape[0], device=temp_values.device), label]

    def __calculate_all_label(self, temp_values):
        return 1 - temp_values