################################################################################################### 
#                               Load modules
###################################################################################################
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cvxpy as cp

import numbers, itertools
from warnings import warn
from abc import ABCMeta, abstractmethod

from functools import partial

from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble._base import BaseEnsemble

from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.utils import indices_to_mask, check_random_state, column_or_1d, deprecated
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._bagging import _parallel_predict_regression, _generate_bagging_indices

from sklearn_misc import _check_sample_weight, _num_samples

MAX_INT = np.iinfo(np.int32).max







################################################################################################### 
#                               Function to fit estimators in parallel
###################################################################################################
def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    sample_weight,
    seeds,
    total_n_estimators,
    verbose,
    check_input,
):


    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_, "sample_weight")
    has_check_input = has_fit_parameter(ensemble.base_estimator_, "check_input")

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append = False, random_state = random_state)

        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input = check_input)
        else:
            estimator_fit = estimator.fit

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator_fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator_fit(X[indices][:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features







################################################################################################### 
#                               Base class for the BaggingRegressor Embedding
###################################################################################################
class BaseBaggingEmbedding(BaseEnsemble, metaclass = ABCMeta):

    @abstractmethod
    def __init__(
        self,
        base_estimator = None,
        n_estimators = 10,
        *,
        max_samples = 1.0,
        max_features = 1.0,
        bootstrap = True,
        bootstrap_features = False,
        oob_score = False,
        warm_start = False,
        n_jobs = None,
        random_state = None,
        verbose = 0,
    ):
        super().__init__(base_estimator = base_estimator, n_estimators = n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose




    def fit(self, X, y, sample_weight = None):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse = ["csr", "csc"],
            dtype = None,
            force_all_finite = False,
            multi_output = True,
        )
        return self._fit(X, y, self.max_samples, sample_weight = sample_weight)



    def _parallel_args(self):
        return {}



    def _fit(
        self,
        X,
        y,
        max_samples = None,
        max_depth = None,
        sample_weight = None,
        check_input = True,
    ):
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype = None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_in_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_in_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size = len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size = n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs = n_jobs, verbose = self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose = self.verbose,
                check_input = check_input,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self



    @abstractmethod
    def _set_oob_score(self, X, y):
        pass


    def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn = True)
        else:
            return y



    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            feature_indices, sample_indices = _generate_bagging_indices(
                seed,
                self.bootstrap_features,
                self.bootstrap,
                self.n_features_in_,
                self._n_samples,
                self._max_features,
                self._max_samples,
            )

            yield feature_indices, sample_indices


    @property
    def estimators_samples_(self):
        return [sample_indices for _, sample_indices in self._get_estimators_indices()]

    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_




################################################################################################### 
#                               Particular class for the BaggingRegressor Embedding
###################################################################################################
class BaggingRegressorEmbedding(RegressorMixin, BaseBaggingEmbedding):

    def __init__(
        self,
        base_estimator = None,
        n_estimators = 10,
        *,
        max_samples = 1.0,
        max_features = 1.0,
        bootstrap = True,
        bootstrap_features = False,
        oob_score = False,
        warm_start = False,
        n_jobs = None,
        random_state = None,
        verbose = 0,
    ):
        super().__init__(
            base_estimator,
            n_estimators = n_estimators,
            max_samples = max_samples,
            max_features = max_features,
            bootstrap = bootstrap,
            bootstrap_features = bootstrap_features,
            oob_score = oob_score,
            warm_start = warm_start,
            n_jobs = n_jobs,
            random_state = random_state,
            verbose = verbose,
        )



    # Possible unsupervised version as in ``RandomTreesEmbedding``
    def fit(self, X, y = None, sample_weight = None):
        if y is None:
            rnd = check_random_state(self.random_state)
            y = rnd.uniform(size = _num_samples(X))
        output = super().fit(X, y, sample_weight = sample_weight)
        return output



    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse = ["csr", "csc"],
            dtype = None,
            force_all_finite = False,
            reset = False,
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_y_hat = Parallel(n_jobs = n_jobs, verbose = self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        
        y_hat = sum(all_y_hat) / self.n_estimators
        return y_hat


    def _validate_estimator(self):
        super()._validate_estimator(default = DecisionTreeRegressor())


    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)




    # Function which gets all of the equations & thresholds from all the estimators for a single sample
    def apply_sample(self, X_sample):
        n_jobs, _, _ = _partition_estimators(len(self.estimators_), self.n_jobs)

        results = Parallel(
            n_jobs = n_jobs,
            verbose = self.verbose,
            prefer = "threads",
        )(delayed(tree.apply_sample)(X_sample, self.estimators_features_[i]) for i, tree in enumerate(self.estimators_))
        
        
        equations = []
        thresholds = []

        # Reduce
        equations += list(itertools.chain.from_iterable(t[0] for t in results))
        thresholds += list(itertools.chain.from_iterable(t[1] for t in results))

        return np.array(equations), np.array(thresholds)
        



    # Functions used in the decoding of a sample
    def decode_sample(self, X_sample, bound_constraints = None):
        sys_equations, sys_thresholds = self.apply_sample(X_sample)
        
        A = np.vstack(sys_equations)
        b = np.hstack(sys_thresholds)

        x = cp.Variable(shape = A.shape[1])

        if bound_constraints:
            constraints = [A @ x >= b, x >= bound_constraints[0], x <= bound_constraints[1]]
        else:
            constraints = [A @ x >= b]
        

        obj = cp.Minimize(cp.norm(x, 1))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        sol = x.value     
        return sol



    # Decoding function
    def decode(self, X, bound_constraints = None):
        n_samples = X.shape[0]
        X_recon = np.zeros_like(X)

        for i in range(n_samples):
            X_recon[i] = self.decode_sample(X[i], bound_constraints)
        return X_recon

        