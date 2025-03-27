from core.learner.learner import Learner
from core.optim.mas import MAS

class MASLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = MAS
        name = "mas"
        super().__init__(name, network, optimizer, optim_kwargs)
