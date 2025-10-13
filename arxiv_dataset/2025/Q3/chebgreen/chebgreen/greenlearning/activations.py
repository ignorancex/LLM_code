from chebgreen.backend import torch, np, config, device

class Rational(torch.nn.Module):
    """
    Rational Activation function.
    f(x) = P(x) / Q(x)
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self, dtype = config(torch), device = device):
        super(Rational, self).__init__()

        self.Pcoeffs = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, 0.0218),
                                     dtype = dtype,
                                     device = device))
        self.Qcoeffs = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype = dtype,
                                     device = device))
        
        self.Pcoeffs.requires_grad_ = True
        self.Qcoeffs.requires_grad_ = True

    def forward(self,
            inputs: torch.Tensor
            ) -> torch.Tensor:
        Num = self.Pcoeffs[3] + inputs*(self.Pcoeffs[2] + inputs*(self.Pcoeffs[1] + inputs*self.Pcoeffs[0]))
        Den = self.Qcoeffs[2] + inputs*(self.Qcoeffs[1] + inputs*self.Qcoeffs[0])
        return torch.div(Num,Den)


def get_activation(identifier):
    """Return the activation function."""
    if isinstance(identifier, str):
        return {
                'elu': torch.nn.ELU(),
                'relu': torch.nn.ReLU(),
                'selu': torch.nn.SELU(),
                'sigmoid': torch.nn.Sigmoid(),
                'tanh': torch.nn.Tanh(),
                'rational': Rational(),
                }[identifier]