import os, sys
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("celoe_clp")[0])
from ontolearn import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.concept_learner import CELOE
from concept_length_predictors.helper_classes import ConceptLengthPredictor
from owlapy.render import DLSyntaxObjectRenderer


class CELOECLP(CELOE):
    def __init__(self, kwargs):
        super().__init__(kwargs['knowledge_base'],
                 kwargs['learning_problem'],
                 kwargs['refinement_operator'],
                 kwargs['quality_func'],
                 kwargs['heuristic_func'],
                 kwargs['terminate_on_goal'],
                 kwargs['iter_bound'],
                 kwargs['max_num_of_concepts_tested'],
                 kwargs['max_runtime'],
                 kwargs['max_results'],
                 kwargs['best_only'],
                 kwargs['calculate_min_max'])
        self.clp = ConceptLengthPredictor(kwargs)
#         self.ignored_concepts = kwargs['ignored_concepts']
        self.renderer = DLSyntaxObjectRenderer()
    
    def result_dict(self, learned_concept=""):
        best_pred = list(self.best_hypotheses(n=1))[0]
        
        d = {"F-measure": best_pred.quality, "Runtime": self.runtime, "Prediction": self.renderer.render(best_pred.concept), "Prediction-Obj": best_pred.concept, "Length": self.kb.cl(best_pred.concept), "Number of Tests":self.number_of_tested_concepts, "Learned Concept": learned_concept}
        return d

        
        
        
        
