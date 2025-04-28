"""
Doubly robust confidence intervals for evaluating/comparing abstaining classifiers
"""

from copy import deepcopy
from typing import Union, Tuple
import numpy as np
import pandas as pd
from collections import OrderedDict

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from mlens.ensemble import SuperLearner

import comparecast as cc
from comparecast_causal.learners import get_learner


def confint_clt(
        xs: np.ndarray,
        alpha: float = 0.05,
        return_all_n: bool = False,
) -> np.ndarray:
    """Compute a two-sided CLT-based asymptotic confidence interval."""
    n = len(xs)
    assert n > 1, "sample size must be >1"

    center = xs.mean()
    var_sq = np.sum((xs - center) ** 2) / (n - 1)
    stderr = np.sqrt(var_sq / n)
    zscore = stats.norm.ppf(1 - alpha / 2)
    return np.array([center - zscore * stderr, center + zscore * stderr])


class DRConfInt:
    """Nonparametric doubly robust confidence intervals for missing-at-random outcomes.

    Handles either:
        1. one pair of (outcomes, missing): estimate the mean of outcomes under missingness
        2. two pairs of (outcomes, missing): estimate the mean *difference* in outcomes under each outcome's missingness
            (when outcomes_b, missing_b are not None)

    Attributes:
        df: a pandas data frame including all inputs, outputs, assignments, and splits
        inputs: a `(N, D)`-shaped numpy array
        outcomes: a `(N, )`-shaped numpy array, representing potentially missing outcomes
        missing: a `(N, )`-shaped boolean numpy array (True: missing, False: observed)
        pi_fn: propensity score learner. Use `:py:func:~comparecast_causal.get_learner()` or set to `None` (default).
        mu0_fn: outcome regressor. Use `:py:func:~comparecast_causal.get_learner()` or set to `None` (default).
        estimator: type of estimator to be used. options: dr (default & recommended), plugin, ipw
        mixed_estimation: whether to use mixed estimation for difference estimates
        cross_fit: whether to perform cross-fitting
        alpha: significance level
        rng: numpy random number generator for shuffling the data

    Methods:
        fit_nuisance_functions
        compute_eifs
        compute_ci
    """

    def __init__(
            self,
            inputs: np.ndarray,
            outcomes: np.ndarray,
            missing: np.ndarray,
            outcomes_b: np.ndarray = None,
            missing_b: np.ndarray = None,
            pi_fn=None,     # lazily defaults to linear
            mu0_fn=None,    # lazily defaults to linear
            clip_pi: float = 0.0,
            estimator: str = "dr",
            mixed_estimation: bool = False,
            mixed_coef: float = None,
            cross_fit: bool = True,
            alpha: float = 0.05,
            assume_iid: bool = True,
            rng: np.random.Generator = np.random.default_rng(),
            verbose: bool = False,
    ):
        self.inputs = inputs
        self.outcomes = outcomes
        self.missing = missing.astype(bool)
        self.outcomes_b = outcomes_b
        self.missing_b = missing_b
        self.is_comparison = self.outcomes_b is not None
        if self.is_comparison and self.missing_b is None:
            raise ValueError("comparison outcome (comparison_b) is given, "
                             "but missingness array (missing_b) is not given")

        self.estimator = estimator.lower()
        self.mixed_estimation = mixed_estimation and self.is_comparison  # not used for evaluation
        self.cross_fit = cross_fit
        self.rng = rng
        self.verbose = verbose

        if self.estimator not in ["dr", "plugin", "ipw"]:
            raise ValueError(f"estimator must be either dr, plugin, or ipw (got: {estimator})")

        if self.missing.all():
            raise ValueError("all predictions are missing; CI cannot be computed")

        self.none_missing = not self.missing.any()
        if self.none_missing:
            raise NotImplementedError("no predictions are missing; proceeding with a standard CI (see `confint_clt`")

        # maintains a length-2 list of nuisance functions for each set of outcomes,
        # depending on which split they are trained on
        self.pi_fn, self.mu0_fn, self.is_nuisance_fit = self._init_nuisance_functions(pi_fn, mu0_fn)
        self.clip_pi = min(clip_pi, 1 - clip_pi)  # clips out large pi predictions
        if self.is_comparison:
            self.pi_fn_b, self.mu0_fn_b, self.is_nuisance_fit_b = self._init_nuisance_functions(pi_fn, mu0_fn)
            if self.mixed_estimation:
                self.pi_fn_diff, self.mu0_fn_diff, self.is_nuisance_fit_diff = self._init_nuisance_functions(pi_fn,
                                                                                                             mu0_fn)
                self.mixed_coef = mixed_coef
        else:
            self.pi_fn_b, self.mu0_fn_b, self.is_nuisance_fit_b = None, None, [False, False]  # placeholder
            self.pi_fn_diff, self.mu0_fn_diff, self.is_nuisance_fit_diff = None, None, [False, False]  # placeholder
            self.mixed_coef = None

        # randomly split data into two halves for cross-fitting
        # for mixed estimation, we leave 10% out for lambda estimation (possibly)
        self.n, self.d = self.inputs.shape
        if self.mixed_estimation and self.mixed_coef is None:
            self.splits = self.rng.multinomial(1, [0.45, 0.45, 0.1], size=self.n).argmax(axis=1)
        else:
            self.splits = self.rng.binomial(1, 0.5, size=self.n)
        self._build_df()

        # CI parameters
        self.alpha = alpha
        self.assume_iid = assume_iid
        assert self.assume_iid, "currently supports IID data only"

        # store results lazily
        self.computed_estimates = False
        self.ci = None
        self.estimate = None

    def _init_nuisance_functions(self, pi_fn, mu0_fn):
        """Initialize pi and mu0."""
        pi_fn = [deepcopy(pi_fn), None]
        mu0_fn = [deepcopy(mu0_fn), None]
        if self.cross_fit:
            pi_fn[1] = deepcopy(pi_fn[0]) if pi_fn is not None else None
            mu0_fn[1] = deepcopy(mu0_fn[0]) if mu0_fn is not None else None
        is_nuisance_fit = [False, False]
        return pi_fn, mu0_fn, is_nuisance_fit

    def _build_df(self):
        """Build a pandas data frame including all inputs, outputs, assignments, and splits."""
        self.input_columns = [f"input{j}" for j in range(1, self.d + 1)]
        self.df = pd.DataFrame(self.inputs, columns=self.input_columns)
        self.df["outcome"] = self.outcomes
        self.df["missing"] = self.missing
        self.df["split"] = self.splits
        self.df["estimate"] = np.nan
        if self.is_comparison:
            self.df["outcome_b"] = self.outcomes_b
            self.df["missing_b"] = self.missing_b

    def _get_split_as_np(self, split=0):
        """select data columns of the data frame in a split and return numpy arrays for each column(s)."""
        columns = [self.input_columns, "outcome", "missing"]
        if self.is_comparison:
            columns += ["outcome_b", "missing_b"]
        else:
            columns += [None, None]
        return [self.df[column][self.df.split == split].to_numpy() if column is not None else None
                for column in columns]

    def fit_nuisance_functions(self, split=0):
        """Fit nuisance functions (`pi_fn`, `mu0_fn`) on `split` (0 or 1)."""
        assert split in {0, 1}, f"split must be either 0 or 1 (got: {split})"
        assert not self.none_missing, "no nuisance functions to be fit, given no missing data"

        if self.is_nuisance_fit[split] or self.is_nuisance_fit_b[split]:
            print("warning: re-fitting nuisance functions that already exist")

        # get inputs
        inputs_tr, outcomes_tr, missing_tr, outcomes_tr_b, missing_tr_b = self._get_split_as_np(split)
        n_tr = len(inputs_tr)

        # fit the missing-neither cases first and exclude from the rest
        if self.mixed_estimation:
            missing_either = np.logical_or(missing_tr, missing_tr_b)  # NOT "00": both known
            if self.estimator in ["plugin", "dr"]:
                if self.mu0_fn_diff[split] is None:
                    self.mu0_fn_diff[split] = get_learner("r", "linear")
                outcomes_diff = outcomes_tr[~missing_either] - outcomes_tr_b[~missing_either]
                self.mu0_fn_diff[split].fit(inputs_tr[~missing_either], outcomes_diff)
            if self.estimator in ["ipw", "dr"]:
                if self.pi_fn_diff[split] is None:
                    self.pi_fn_diff[split] = get_learner("c", "linear")
                self.pi_fn_diff[split].fit(inputs_tr, missing_either)
            self.is_nuisance_fit_diff[split] = True

            # exclude the cases from the later calculations
            assert missing_either.any(), "no missing data!"
            # m_a, m_b = missing_tr, missing_tr_b
            # missing_tr = ~np.logical_and(~m_a, m_b)    # NOT "01": also exclude if b was known
            # missing_tr_b = ~np.logical_and(m_a, ~m_b)  # NOT "10": also exclude if a was known

        # fit outcome regressors (mu0) for plugin or dr
        if self.estimator in ["plugin", "dr"]:
            if self.mu0_fn[split] is None:
                self.mu0_fn[split] = get_learner("r", "linear")
            self.mu0_fn[split].fit(inputs_tr[~missing_tr], outcomes_tr[~missing_tr])
            if self.is_comparison:
                if self.mu0_fn_b[split] is None:
                    self.mu0_fn_b[split] = get_learner("r", "linear")
                self.mu0_fn_b[split].fit(inputs_tr[~missing_tr_b], outcomes_tr_b[~missing_tr_b])

        # fit propensity scores (pi) for ipw or dr
        if self.estimator in ["ipw", "dr"]:
            if self.pi_fn[split] is None:
                self.pi_fn[split] = get_learner("c", "linear")
            self.pi_fn[split].fit(inputs_tr, missing_tr)
            if self.is_comparison:
                if self.pi_fn_b[split] is None:
                    self.pi_fn_b[split] = get_learner("c", "linear")
                self.pi_fn_b[split].fit(inputs_tr, missing_tr_b)

        self.is_nuisance_fit[split] = True
        if self.is_comparison:
            self.is_nuisance_fit_b[split] = True

        if self.verbose:
            self.evaluate_nuisance_functions(split=split, eval_split=1 - split)

    def evaluate_nuisance_functions(self, split=0, eval_split=1):
        """Evaluate the training of nuisance functions on the eval split."""
        assert self.is_nuisance_fit[split], "tried evaluating nuisance functions that were not learned yet"

        # evaluation data
        inputs, outcomes, missing, outcomes_b, missing_b = self._get_split_as_np(eval_split)

        nfs = OrderedDict({
            "pi": (self.pi_fn[split], inputs, missing),
            "mu0": (self.mu0_fn[split], inputs[~missing], outcomes[~missing]),
        })
        if self.is_comparison:
            nfs.update({
                "pi_b": (self.pi_fn_b[split], inputs, missing_b),
                "mu0_b": (self.mu0_fn_b[split], inputs[~missing_b], outcomes_b[~missing_b]),
            })
        if self.mixed_estimation:
            missing_either = np.logical_or(missing, missing_b)
            nfs.update({
                "pi_diff": (self.pi_fn_diff[split], inputs, missing_either),
                "mu0_diff": (self.mu0_fn_diff[split], inputs[~missing_either],
                             outcomes[~missing_either] - outcomes_b[~missing_either]),
            })

        # remove invalid nfs
        if self.estimator == "plugin":
            nfs = OrderedDict({k: v for k, v in nfs.items() if k.startswith("mu0")})
        if self.estimator == "ipw":
            nfs = OrderedDict({k: v for k, v in nfs.items() if k.startswith("pi")})

        scores = OrderedDict()
        for nf_name, (nf, inp, out) in nfs.items():
            score_fn = accuracy_score if nf_name.startswith("pi") else mean_squared_error
            # mlens
            if isinstance(nf, SuperLearner):
                pred = nf.predict(inp)
                if nf_name.startswith("pi") and len(pred.shape) > len(out.shape):  # classification
                    pred = pred.argmax(axis=-1)
                scores[nf_name] = score_fn(out, pred)
            # scikit-learn
            else:
                scores[nf_name] = score_fn(out, nf.predict(inp))

            if self.verbose:
                print("{} [split={}] evaluation: {} {:.5f}".format(
                    nf_name, split, "accuracy" if nf_name.startswith("pi") else "MSE", scores[nf_name]))
        return scores

    def _compute_ipw(self, inputs, missing, outcomes, pi_fn, mu0_preds=None):
        """A helper for ipw estimator computation."""
        # TODO(yj): check for getting the correct probability with SuperLearners
        pi_preds = pi_fn.predict_proba(inputs)[:, 1].clip(0, 1 - self.clip_pi)
        # abst_index = np.argmax(self.pi_fn[split].classes_)  # gets the index for 'True'
        # pi_preds = pi_preds[:, abst_index]
        observed = ~missing
        ipw = 1.0 / np.maximum(1e-5, 1 - pi_preds[observed])
        outcomes_ipw = (outcomes[observed] if self.estimator == "ipw" else outcomes[observed] - mu0_preds[observed])
        return ipw * outcomes_ipw

    def _compute_estimates_on_half(self, split=0, eval_split=1):
        """Compute the pointwise (EIF) estimates on `eval_split` using nuisance functions trained on `split`.
        """
        assert self.is_nuisance_fit[split] and (not self.is_comparison or self.is_nuisance_fit_b[split])

        inputs_eval, outcomes_eval, missing_eval, outcomes_eval_b, missing_eval_b = self._get_split_as_np(eval_split)
        n_eval = len(inputs_eval)

        # take out the neither-abstention cases and fit them first
        if self.mixed_estimation:
            mixed_estimates = np.zeros(n_eval)
            missing_either = np.logical_or(missing_eval, missing_eval_b)  # NOT "00": both known
            if self.estimator in ["plugin", "dr"]:
                mu0_preds_diff = self.mu0_fn_diff[split].predict(inputs_eval)
                mixed_estimates += mu0_preds_diff
            else:
                mu0_preds_diff = None
            if self.estimator in ["ipw", "dr"]:
                outcomes_diff = outcomes_eval - outcomes_eval_b
                mixed_estimates[~missing_either] += self._compute_ipw(inputs_eval, missing_either, outcomes_diff,
                                                                      self.pi_fn_diff[split], mu0_preds_diff)
            # exclude the cases from the later ipw calculations
            assert missing_either.any(), "no missing data!"
            # m_a, m_b = missing_eval, missing_eval_b
            # missing_eval = np.logical_or(m_a, ~m_b)    # NOT "01": also exclude if b was known
            # missing_eval_b = np.logical_or(~m_a, m_b)  # NOT "10": also exclude if a was known
        else:
            mixed_estimates = None

        estimates = np.zeros(n_eval)
        mu0_preds, mu0_preds_b = None, None
        if self.estimator in ["plugin", "dr"]:
            mu0_preds = self.mu0_fn[split].predict(inputs_eval)
            estimates += mu0_preds
            # Delta^AB = psi^A - psi^B
            if self.is_comparison:
                mu0_preds_b = self.mu0_fn_b[split].predict(inputs_eval)
                estimates -= mu0_preds_b

        if self.estimator in ["ipw", "dr"]:
            estimates[~missing_eval] += self._compute_ipw(inputs_eval, missing_eval, outcomes_eval,
                                                          self.pi_fn[split], mu0_preds)
            # Delta^AB = psi^A - psi^B
            if self.is_comparison:
                estimates[~missing_eval_b] -= self._compute_ipw(inputs_eval, missing_eval_b, outcomes_eval_b,
                                                                self.pi_fn_b[split], mu0_preds_b)

        return estimates, mixed_estimates

    def _find_lambda(self, split):
        """Find the optimal estimate of lambda using the nuisance functions trained on 'split'."""
        assert self.mixed_estimation
        assert split in [0, 1], f"split must be 0 or 1, got {split}"
        assert self.is_nuisance_fit[split] and self.is_nuisance_fit_b[split] and self.is_nuisance_fit_diff[split]

        # find optimal lambda on the "lambda split"
        estimates, mixed_estimates = self._compute_estimates_on_half(split, eval_split=2)
        v_mixed = mixed_estimates.var()
        v_diff = estimates.var()
        mixed_coef = v_mixed / np.maximum(1e-5, v_mixed + v_diff)
        if self.verbose:
            print(f"means: {mixed_estimates.mean():.5f}, {estimates.mean():.5f}")
            print(f"variances: {v_mixed:.5f}, {v_diff:.5f} (coef: {mixed_coef})")
        return mixed_coef

    def compute_estimates(self):
        """Compute the pointwise estimates of the target parameter on the evaluation set.

        For the doubly robust estimator, this computes the efficient influence functions (EIFs).

        If NOT `self.cross_fit`, then the resulting array will have missing points (the nuisance split).
        """
        # one half for training nuisance functions; other half for evaluation
        for split in [0, 1]:
            if not self.cross_fit and split == 1:
                break

            eval_split = 1 - split
            if not self.is_nuisance_fit[split]:
                self.fit_nuisance_functions(split=split)
            est, mixed_est = self._compute_estimates_on_half(split, eval_split)
            if self.mixed_estimation:
                mixed_coef = self.mixed_coef if self.mixed_coef is not None else self._find_lambda(split)
                self.df.loc[self.df.split == eval_split, "estimate"] = (1 - mixed_coef) * mixed_est + mixed_coef * est
            else:
                self.df.loc[self.df.split == eval_split, "estimate"] = est

        self.estimate = self.df.estimate.mean()
        self.computed_estimates = True
        return self.df.estimate.to_numpy()

    def compute_ci(self):
        """Compute the asymptotic CI."""
        if not self.computed_estimates:
            self.compute_estimates()

        # only take available estimates (not nans)
        self.ci = confint_clt(self.df.estimate.dropna().to_numpy(), self.alpha)
        return self.ci


class DREvalAbst(DRConfInt):
    """Doubly robust confidence intervals for evaluating/classifying abstaining classifiers.

    Estimates the (difference in) counterfactual score of the classifier under MAR and positivity assumptions.

    We use the fact that EIFs for each classifier can be simply added up to obtain the difference EIF.

    Attributes:
        inputs: a `(N, D)`-shaped numpy array
        labels: a `(N, )`-shaped numpy array of labels
        predictions: `(N, C)`-shaped numpy array(s) containing probability predictions for the label.
            (Order: A, B -> estimated quantity is s(p_A, y) - s(p_B, y).)
        abstentions: `(N, )`-shaped numpy array(s) containing binary abstention decisions.
            (Order: same as `predictions`)
        scoring_rule: name of the scoring rule used (default: brier)
        estimator: type of estimator to be used. options: dr (default & recommended), plugin, ipw
    """

    def __init__(
            self,
            inputs: np.ndarray,
            labels: np.ndarray,
            predictions: np.ndarray,
            abstentions: np.ndarray,
            predictions_b: np.ndarray = None,
            abstentions_b: np.ndarray = None,
            scoring_rule: Union[str, cc.ScoringRule] = "brier",
            pi_fn=None,
            mu0_fn=None,
            clip_pi: float = 0.0,
            estimator: str = "dr",
            mixed_estimation: bool = True,
            mixed_coef: float = 0.5,
            cross_fit: bool = True,
            alpha: float = 0.05,
            assume_iid: bool = True,
            rng: np.random.Generator = np.random.default_rng(),
            verbose: bool = False,
    ):
        self.inputs = inputs
        self.labels = labels
        self.predictions = predictions
        self.abstentions = abstentions
        self.predictions_b = predictions_b
        self.abstentions_b = abstentions_b
        self.scoring_rule = cc.get_scoring_rule(scoring_rule)
        self.rng = rng

        self._preprocess_nuisance_outputs()
        super().__init__(
            inputs=self.inputs,
            outcomes=self.scores,
            missing=self.missing,
            outcomes_b=self.scores_b,
            missing_b=self.missing_b,
            pi_fn=pi_fn,
            mu0_fn=mu0_fn,
            clip_pi=clip_pi,
            estimator=estimator,
            mixed_estimation=mixed_estimation,
            mixed_coef=mixed_coef,
            cross_fit=cross_fit,
            alpha=alpha,
            assume_iid=assume_iid,
            rng=rng,
            verbose=verbose,
        )

    def _preprocess_nuisance_outputs(self):
        """preprocess abstentions & scores, which are the outputs for nuisance functions."""
        n = len(self.inputs)
        assert len(self.predictions) == len(self.abstentions) == n

        n, c = self.predictions.shape
        assert len(self.inputs) == len(self.labels) == len(self.abstentions) == n
        assert set(self.labels).issubset({i for i in range(c)})

        self.scores = np.zeros(n)
        self.missing = self.abstentions.astype(bool)
        self.scores[~self.missing] = self.scoring_rule(self.predictions[~self.missing],
                                                       self.labels[~self.missing])

        # comparison
        if self.predictions_b is not None:
            assert len(self.predictions_b) == len(self.abstentions_b) == n
            assert self.predictions_b.shape == self.predictions.shape
            assert self.abstentions_b.shape == self.abstentions.shape

            self.scores_b = np.zeros(n)
            self.missing_b = self.abstentions_b.astype(bool)
            self.scores_b[~self.missing_b] = self.scoring_rule(self.predictions_b[~self.missing_b],
                                                               self.labels[~self.missing_b])
        else:
            self.scores_b = None
            self.missing_b = None
