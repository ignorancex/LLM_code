from models.compositors.MLP import Linear, Adapter, Projector, Phi
from models.compositors.Combiner import Combiner
from torch import nn

def compositors_factory(input_dim: int, configs: dict, output_dim: int = None):
    """
    :param input_dim: clip feature output dimension
    """
    model_code = configs['compositor']
    if model_code == Linear.code():
        # assume input_dim == output_dim
        return Linear(input_dim, input_dim // 2, output_dim if output_dim else input_dim)
    if model_code == 'adaptor':
        return Adapter(input_dim)
    if model_code == Projector.code():
        return Projector(input_dim, output_dim=256)
    if model_code == Combiner.code():
        return Combiner(clip_feature_dim=input_dim, hidden_dim=input_dim*8, projection_dim=input_dim*4)
    if model_code == Phi.code():
        return Phi(input_dim, hidden_dim=input_dim, output_dim=output_dim)
    Warning("There's no compositor matched with {}".format(model_code))
    # return identity with code 'identity'
    class Identity(nn.Module):
        @classmethod
        def code(cls):
            return 'identity'
        def forward(self, x):
            return x
    return Identity()