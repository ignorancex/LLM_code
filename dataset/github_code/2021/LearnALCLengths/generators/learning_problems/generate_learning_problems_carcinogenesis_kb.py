import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('learning_problems')[0])

from learning_problem_generator import LearningProblemGenerator
from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement

kb_path = os.path.dirname(os.path.realpath(__file__)).split('generators')[0]+"Datasets/carcinogenesis/carcinogenesis.owl"
kb = KnowledgeBase(path=kb_path)
lpg = LearningProblemGenerator(kb_path=kb_path, depth=2, num_rand_samples=50, max_num_probs_per_length=50, max_ref_child_length=5, refinement_expressivity=0.1, min_num_pos_examples=1, max_num_pos_examples=2000)
lpg.Filter().save_learning_problems()