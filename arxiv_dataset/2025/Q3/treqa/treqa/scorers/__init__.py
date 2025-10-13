import sys

from .chrf_scorer import ChrfScorer
from .treqa_scorer import TREQAScorer
from .treqa_qe_scorer import TREQAQEScorer
from .keyphrase_scorer import KeyPhraseScorer
from .gemba_scorer import GembaScorer


METRICS_REGISTRY = {
    "chrf": ChrfScorer,
    "treqa": TREQAScorer,
    "treqa_qe": TREQAQEScorer,
    "keyphrase": KeyPhraseScorer,
    "gemba": GembaScorer,
}

try:
    from .comet_scorer import CometScorer
    from .cometqe_scorer import CometQEScorer

    METRICS_REGISTRY.update(
        {
            "comet": CometScorer,
            "cometqe": CometQEScorer,
        }
    )
except ImportError:
    print(
        "WARNING: Comet is not installed, so its scorers will not be available.",
        file=sys.stderr,
    )

try:
    from .metricx_scorer import MetricXScorer, MetricXQEScorer

    METRICS_REGISTRY.update(
        {
            "metricx": MetricXScorer,
            "metricx_qe": MetricXQEScorer,
        }
    )
except ImportError:
    print(
        "WARNING: MetricX is not installed, so its scorers will not be available.",
        file=sys.stderr,
    )
