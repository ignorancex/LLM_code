from .normal import NormalLearner
from .hgg import HGGLearner
from .normal_goalGAN import NormalGoalGANLearner
from .maxq import MaxQLearner
from .pickmaxq import PickMaxQLearner
from .pickthrowmaxq import PickThrowMaxQLearner
from .handmaxq import HandMaxQLearner
from .multi_step import MultiStepLearner

learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
	'normal+goalGAN': NormalGoalGANLearner,
    'maxq': MaxQLearner,
    'pickmaxq': PickMaxQLearner, 
    'pickthrowmaxq': PickThrowMaxQLearner,
    'handmaxq': HandMaxQLearner,
    'multistep': MultiStepLearner
}

def create_learner(args):
	return learner_collection[args.learn](args)
