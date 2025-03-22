
class Explainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.explainer = None
    
    def explain(self, idx, **kwargs):
        pass
