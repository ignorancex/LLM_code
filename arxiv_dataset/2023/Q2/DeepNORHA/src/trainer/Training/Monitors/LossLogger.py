from .MetricLogger import MetricLogger

class LossLogger(MetricLogger):
    def __init__(self, **args) -> None:
        super().__init__(metric="Loss", **args)