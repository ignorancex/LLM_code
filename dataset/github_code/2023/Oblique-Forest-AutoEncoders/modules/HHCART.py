################################################################################################### 
#                               Load modules
###################################################################################################
import numpy as np

from copy import deepcopy
from scipy.linalg import norm

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import Ridge





################################################################################################### 
#                               HHCART node class
###################################################################################################
class HHCART_Node:

    # Constructor
    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        
        self._left_child = kwargs.get("left_child", None)
        self._right_child = kwargs.get("right_child", None)

        self.is_leaf = kwargs.get("is_leaf", False)
        
        self._split_rules = kwargs.get("split_rules", None)
        self._weights = kwargs.get("weights", None)


        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child
        
        
    

    # Function which gets the child of the current node by using an input sample
    def get_child(self, sample):
        
        if self.is_leaf:
            raise Exception("Leaf node does not have children !")

        X = deepcopy(sample)

        if X.dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child


    
    # Properties to be used in external code
    @property
    def label(self):
        if not hasattr(self, "_label"):
            self._label = np.mean(self.labels)
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule !")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule !")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule !")
        return self._right_child






################################################################################################### 
#                               HHCART tree estimator class
###################################################################################################
class HouseHolder_CART(BaseEstimator):

    # Constructor
    def __init__(self, impurity, segmentor, method = "eig", tau = 1e-4, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self.method = method
        self.tau = tau

        self._max_depth = kwargs.get("max_depth", None)
        self._min_samples = kwargs.get("min_samples", 2)

        self._root = None
        self._nodes = []




    # Function which checks that the tree construction should finish
    def _terminate(self, X, y, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            return True

        elif y.size < self._min_samples:
            return True

        elif np.unique(y).size == 1:
            return True

        else:
            return False


    
    # Function which generates leaf nodes
    def _generate_leaf_node(self, cur_depth, y):
        node = HHCART_Node(cur_depth, y, is_leaf = True)
        self._nodes.append(node)
        return node



    # Function which constructs recursively internal & leaf nodes
    def _generate_node(self, X, y, cur_depth):

        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        
        else:
            n_features = X.shape[1]
            impurity_best, sr, left_indices, right_indices = self.segmentor(X, y, self.impurity)
            oblique_split = False


            # Generate Housholder matrix
            if self.method == "eig":
                extractor = PCA(n_components = 1)
                extractor.fit(X, y)
                mu = extractor.components_[0]

            if self.method == "svd":
                extractor = TruncatedSVD(n_components = 1)
                extractor.fit(X, y)
                mu = extractor.components_[0]

            if self.method == "fast_ica":
                extractor = FastICA(n_components = 1)
                extractor.fit(X, y)
                mu = extractor.components_[0]

            if self.method == "factor":
                extractor = FactorAnalysis(n_components = 1)
                extractor.fit(X, y)
                mu = extractor.components_[0]

            if self.method == "proj":
                extractor = GaussianRandomProjection(n_components = 1)
                extractor.fit(X, y)
                mu = extractor.components_[0]


            I = np.diag(np.ones(n_features))
            check_ = np.sqrt(((I - mu) ** 2).sum(axis = 1))

            if (check_ > self.tau).sum() > 0:
                i = np.argmax(check_)
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu) / norm(e - mu)
                householder_matrix = I - 2 * w[:, np.newaxis].dot(w[:, np.newaxis].T)

                X_house = X.dot(householder_matrix)

                (
                    impurity_house,
                    sr_house,
                    left_indices_house,
                    right_indices_house,
                ) = self.segmentor(X_house, y, self.impurity)

                if impurity_best > impurity_house:
                    oblique_split = True
                    impurity_best = impurity_house
                    left_indices = left_indices_house
                    right_indices = right_indices_house
                    sr = sr_house
            else:
                householder_matrix = I

            if not sr:
                return self._generate_leaf_node(cur_depth, y)

            i, treshold = sr
            weights = np.zeros(n_features + 1)
            
            if oblique_split:
                weights[:-1] = householder_matrix[:, i]
            else:
                weights[i] = 1
                
            weights[-1] = treshold

            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            node = HHCART_Node(
                cur_depth,
                y,
                split_rules = sr,
                weights = weights,
                left_child = self._generate_node(X_left, y_left, cur_depth + 1),
                right_child = self._generate_node(X_right, y_right, cur_depth + 1),
                is_leaf = False,
            )

            self._nodes.append(node)
            return node



    
    # Train function
    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)



    # Function which predicts the targets for a single sample
    def predict_single(self, sample):
        cur_node = self._root
        while not cur_node.is_leaf:
            cur_node = cur_node.get_child(sample)
        return cur_node.label




    # Function which predicts the targets for multiple samples
    def predict(self, X):
        if not self._root:
            raise Exception("Decision tree has not been trained !")

        size = X.shape[0]
        predictions = np.empty((size, ), dtype = float)

        for i in range(size):
            predictions[i] = self.predict_single(X[i, :])
        return predictions
    
    
    
    
    # Function which takes the equations for the given sample
    def apply_sample(self, sample, features_):
        cur_node = self._root
        equations = []
        thresholds = []
        
        while not cur_node.is_leaf:
            if cur_node.is_leaf:
                raise Exception("Leaf node does not have children !")
            
            X = deepcopy(sample)
            total_n_features_ = len(range(X.shape[0]))
            X = np.take(X, indices = features_)
            
            if X.dot(np.array(cur_node._weights[:-1]).T) < cur_node._weights[-1]:
                equations.append(-cur_node._weights[:-1])
                thresholds.append(-cur_node._weights[-1])
                cur_node = cur_node.left_child
            else:
                equations.append(cur_node._weights[:-1])
                thresholds.append(cur_node._weights[-1])
                cur_node = cur_node.right_child

            temp = np.zeros(shape = total_n_features_)
            temp[features_] = equations[-1]
            equations[-1] = temp

        return equations, thresholds




