from lime.lime_tabular import LimeTabularExplainer
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from posthoc_explanations.types import Explainer

class Lime(Explainer):
    # mode: "classification" or "regression"
    def __init__(self, full_X, pred_func, feature_names, mode="regression"):
        super().__init__(full_X, pred_func)
        self.pred_func=pred_func
        self.explainer = LimeTabularExplainer(full_X, feature_names=feature_names, discretize_continuous=False, mode=mode)
    
    def explain(self, x, num_features=10, num_samples=20):
        return self.explainer.explain_instance(x, self.pred_func, num_features=num_features, num_samples=num_samples)
        
