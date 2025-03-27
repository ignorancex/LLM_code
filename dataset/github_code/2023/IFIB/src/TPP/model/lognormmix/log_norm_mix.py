import torch
import torch.nn as nn

import torch.distributions as D
from src.TPP.model.lognormmix.distributions import Normal, MixtureSameFamily, TransformedDistribution
from src.TPP.model.lognormmix.utils import clamp_preserve_gradients

from src.TPP.model.lognormmix.recurrent_tpp import RecurrentTPP


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,                                                    # \mu
        log_scales: torch.Tensor,                                              # s without exp()
        log_weights: torch.Tensor,                                             # w
    ):
        mixture_dist = D.Categorical(logits = log_weights)
        component_dist = Normal(loc = locs, scale = log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        transforms = [D.ExpTransform(),]
        super().__init__(GMM, transforms)


    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()


class LogNormMix(RecurrentTPP):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
        self,
        num_marks: int,
        device: str,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        num_mix_components: int = 16,
        rnn_type: str = "GRU",
    ):
        super().__init__(
            num_marks=num_marks,
            context_size=context_size,
            mark_embedding_size=mark_embedding_size,
            rnn_type=rnn_type,
            device = device
        )
        self.device = device
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components, device = self.device)


    def get_inter_time_dist(self, context_history: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context_history)                              # [batch_size, seq_len, 3 * num_mix_components]
        # Slice the tensor to get the parameters of the mixture
        locs, log_scales, log_weights = torch.chunk(raw_params, 3, dim = -1)   # 3 * [batch_size, seq_len, num_mix_components]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim = -1)

        return LogNormalMixtureDistribution(
            locs = locs,
            log_scales = log_scales,
            log_weights = log_weights
        )
