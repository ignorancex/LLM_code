import numpy as np
import eif as iso

class ExtendedIsoForest():
    def __init__(self, ntrees=200, sample_size=256):
        self.forest = None
        self.ntrees = 200
        self.sample_size = 256
    
    def fit(self, X):
        X = X.astype(np.float64)
        self.forest = iso.iForest(X, ntrees=self.ntrees,
            sample_size=self.sample_size, ExtensionLevel=1)
        return self
    
    def predict(self, X):
        X = X.astype(np.float64)
        return self.score_samples(X)

    def score_samples(self, X):
        X = X.astype(np.float64)
        return self.forest.compute_paths(X_in=X)
