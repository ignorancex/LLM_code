from typing import List, Tuple, Dict, Any, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from copy import deepcopy

from . import utils

class EmbeddingsRegressor(BaseEstimator, RegressorMixin):

    def __init__(
            self, 
            encoder: Union[SentenceTransformer, AutoTokenizer], 
            regressor: Union[MultiOutputRegressor, BaseEstimator],
            normalize_output: bool = True,
            verbose: bool = False
        ):
        self.encoder = encoder
        self.regressor = regressor
        self.normalize_output = normalize_output
        self.encodings = None
        self.verbose = verbose

    def fit(self, X: List[str], y: List[Tuple[float, float, float, float]]) -> "EmbeddingsRegressor":
        X = self.encoder.encode(X, show_progress_bar=self.verbose)
        self.regressor.fit(X, y)
        return self
    
    def transform(self, X: List[str]) -> List[List[float]]:
        X = self.encoder.encode(X, show_progress_bar=self.verbose)
        self.encodings = X
        return X
    
    def predict(self, X: Union[List[str], np.array], encodings=False) -> Union[List[float],List[List[float]]]:
        if not encodings:
            X = self.encoder.encode(X, show_progress_bar=self.verbose)
        self.encodings = X
        pred = self.regressor.predict(X)
        if self.normalize_output:
            pred /= pred.sum(axis=1, keepdims=True)
        return pred
        