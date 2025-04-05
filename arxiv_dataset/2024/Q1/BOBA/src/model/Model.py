import torch
import torch.nn as nn
from .utils import tensor_to_state, state_to_tensor


class Model(nn.Module):
    """
    Model base class
    """

    def uploaded_state_dict(self):
        """
        Parameters that are uploaded to the server for aggregation
        By default, it is all the parameters
        In GRU, it is the parameter except for the frozen embedding layer
        """
        return self.state_dict()

    def get_params_tensor(self):
        """
        Extract the parameters for aggregation, as a d-dimensional vector
        """
        state = self.uploaded_state_dict()
        tensor = state_to_tensor(state)
        return tensor

    def load_params_tensor(self, tensor):
        """
        Load a d-dimensional vector as the parameter of the model
        Only the "uploaded" parameters a overwritten
        """
        template = self.uploaded_state_dict()
        new_state = tensor_to_state(tensor, template)
        self.load_state_dict(new_state, strict=False)

    def trainable_parameter_tensor(self):
        """
        Get trainable parameter as a tensor, support back propagation
        """
        trainable_parameters = [tensor.view(-1) for tensor in self.parameters() if tensor.requires_grad]
        return torch.cat(trainable_parameters)


def test():
    model = Model()
    print(model.state_dict())  # OrderedDict()
    print(model.uploaded_state_dict())  # OrderedDict()
