"""Multivariate Student T distribution."""
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from simanneal import Annealer

#from psych_metric.supervised_joint_distrib import is_prob_distrib


class MultivariateStudentT(object):
    """The Multivariate Student T distribution.

    Attributes
    ----------
    df : int
        The degrees of freedom
    loc : np.ndarray(float)
        The means of the different dimensions experessed as a vector of postive
        floats.
    sigma : np.ndarray(float)
        The sigma matrix of the multivariate student T distribution. This is
        NOT the scale matrix! This is equal to scale @ scale.T
    const : list | set
        List of parameter names that are constant.
    """

    def __init__(self, df, loc, sigma, scale=True):
        if not isinstance(df, float):
            raise TypeError(f'`df` must be of type float, not{type(df)}')
        if not isinstance(loc, np.ndarray):
            raise TypeError(f'`loc` must be of type np.ndarray, not {type(loc)}')
        if loc.dtype != np.float:
            raise TypeError(f'`loc` elements must be of type float, not {loc.dtype}')
        if not isinstance(sigma, np.ndarray):
            raise TypeError(f'`sigma` must be of type np.ndarray, not {type(df)}')
        if sigma.dtype != np.float:
            raise TypeError(f'`sigma` elements must be of type float, not {sigma.dtype}')
        # TODO check if scale is a valid positive definite matrix

        self.df = df
        self.loc = loc
        self.sigma = sigma @ sigma.T if scale else sigma

    def fit(self, data, method='nelder_mead', *args, **kwargs):
        if method == 'nelder_mead':
            return self.nelder_mead(data, *args, **kwargs)

        raise ValueError(f'Only the Nedler-Mead method is supported, not `{method}`.')

    def log_prob(self, samples):
        return self._log_prob(samples, self.df, self.loc, self.sigma)

    def _log_prob(self, samples, df, loc, sigma):
        if not isinstance(samples, np.ndarray):
            try:
                samples = np.array(samples)
            except:
                raise(f'`samples` is expected to convertible to a numpy array.')

        if len(samples.shape) == 2 and samples.shape[1] == len(loc):
            # Handle multiple samples
            return np.array([
                self._log_probability(x, df, loc, sigma)
                for x in samples
            ])
        elif len(samples.shape) == 1 and len(samples) == len(loc):
            # Handle the single sample case
            return self._log_probability(samples, df, loc, sigma)
        else:
            raise TypeError(' '.join([
                'Expected given `samples` to be of either', 'dimension 2 when',
                'multiple samples (with the rows as samples) or 1 when a',
                'single sample. The samples must have the same dimensions as',
                '`loc`.',
            ]))

    def _log_probability(self, sample, df, loc, sigma):
        """Returns the log probability of a single sample."""
        dims = len(loc)
        return (
            scipy.special.gammaln((df + dims) / 2)
            - (df + dims) / 2 * (
                1 + (1 / df) * (sample - loc) @ np.linalg.inv(sigma)
                @ (sample - loc).T
            ) - (
                scipy.special.gammaln(df / 2)
                + .5 * (
                    dims * (np.log(df) + np.log(np.pi))
                    + np.log(np.linalg.norm(sigma))
                )
            )
        )

    def sample(self, num_samples):
        """Sample from the estimated joint probability distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the distribution.

        Returns
        -------
        (np.ndarray, np.ndarray), shape(samples, input_dim, 2)
            Two arrays of samples from the joint probability distribution of
            the target and the predictor output aligned by samples in their
            first dimension.
        """
        sess = tf.Session()

        lower_tri = tf.linalg.LinearOperatorLowerTriangular(self.sigma)

        tfp_mvst = tfp.distributions.MultivariateStudentTLinearOperator(
            self.df,
            self.loc,
            lower_tri,
        )

        samples = tfp_mvst.sample(num_samples).eval(session=sess)

        # NOTE using Tensorflow while loop to resample and check/rejection
        # would be more optimal. This is next step if necessary.

        """
        # Get any and all indices of non-probability distrib samples
        bad_sample_idx = np.argwhere(np.logical_not(
            is_prob_distrib(samples),
        ))
        if len(bad_sample_idx) > 1:
            bad_sample_idx = np.squeeze(bad_sample_idx)
        num_bad_samples = len(bad_sample_idx)
        #adjust_bad = num_bad_samples * (num_samples / (num_samples - num_bad_samples))

        while num_bad_samples > 0:
            print(f'Bad Times: num bad samples = {num_bad_samples}')

            # rerun session w/ enough samples to replace bad samples and some.
            new_pred = tfp_mvst.sample(num_bad_samples).eval(session=sess)

            samples[bad_sample_idx] = new_pred

            bad_sample_idx = np.argwhere(np.logical_not(
                is_prob_distrib(samples),
            ))
            if len(bad_sample_idx) > 1:
                bad_sample_idx = np.squeeze(bad_sample_idx)
            num_bad_samples = len(bad_sample_idx)
        #"""

        return  samples

    def mvst_neg_log_prob(
        self,
        x,
        data,
        constraint_multiplier=1e5,
        const=None,
    ):
        """the Log probability density function of multivariate

        mean is kept constant to the datas column means.
        """
        dims = data.shape[1]

        # expand the params into their proper forms
        if isinstance(const, dict) and 'df' in const:
            df = const['df']
            loc = x[: dims]
            scale = np.reshape(x[dims:], [dims, dims])
        else:
            df = x[0]
            loc = x[1 : dims + 1]
            scale = np.reshape(x[dims + 1:], [dims, dims])

        # Get Sigma Matrix from scale.
        sigma = scale @ scale.T

        # Get the negative log probability of the data
        loss = - self._log_prob(data, df, loc, sigma).sum()

        # apply constraints to variables
        if not (isinstance(const, dict) and 'df' in const) and df <= 0:
            loss += (1e-4 - df) * constraint_multiplier

        return loss

    def nelder_mead(
        self,
        data,
        const=None,
        max_iter=10000,
        nelder_mead_args=None,
        name='nelder_mead_multivarite_student_t',
        constraint_multiplier=1e5,
    ):
        """Estimates the Maximum Likelihood Estimated of the Multivariate Student
        using Nelder-Mead optimization to minimize the negative log-likelihood
        """
        if nelder_mead_args is None:
            optimizer_args = {}

        if const and 'df' in const:
            print(f'df is const. df = {self.df}.')

            init_data = np.concatenate([self.loc, self.sigma.flatten()])
            return scipy.optimize.minimize(
                lambda x: self.mvst_neg_log_prob(x, data, const={'df': self.df}),
                init_data,
                method='Nelder-Mead',
                options={'maxiter': max_iter},
            )
        else:
            init_data = np.concatenate([[self.df], self.loc, self.sigma.flatten()])
            return scipy.optimize.minimize(
                lambda x: self.mvst_neg_log_prob(x, data),
                init_data,
                method='Nelder-Mead',
                options={'maxiter': max_iter},
            )

    def simulated_anneal(self, data, const=None, auto_min=1, processes=8):
        sa_mvst = MVSTSimAnneal(data, self, const)
        sa_mvst.set_schedule(sa_mvst.auto(minutes=auto_min))
        sa_mvst.copy_strategy = 'deepcopy'

        return sa_mvst.anneal()
        """
        with Pool(processes=processes) as pool:
            states = pool.starmap(
                anneal,
                zip(
                    [data] * processes,
                    [self] * processes,
                    [const] * processes,
                ),
            )
        return states
        """


def anneal(data, mvst, const, auto_min=1):
    sa_mvst = MVSTSimAnneal(data, mvst, const)
    sa_mvst.set_schedule(sa_mvst.auto(minutes=auto_min))
    sa_mvst.copy_strategy = 'deepcopy'

    return sa_mvst.anneal()


class MVSTSimAnneal(Annealer):
    """Simulated annealing of the Multivariate Student T's parameters."""

    def __init__(self, data, mvst, const=None, dtype=np.float32):
        self.data = data
        self.mvst = mvst
        self.const = const if const is not None else []
        self.dtype = dtype

        state = {
            'df': mvst.df,
            'loc': mvst.loc,
            'sigma': mvst.sigma,
        }

        super(MVSTSimAnneal, self).__init__(state)

    def move(self):
        if 'df' not in self.const:
            self.state['df'] = np.random.exponential(5)

        if 'loc' not in self.const:
            self.state['loc'] = np.random.uniform(
                np.finfo(self.dtype).tiny,
                np.finfo(self.dtype).max,
                len(self.mvst.loc),
            )

        if 'sigma' not in self.const:
            self.state['sigma'] = np.random.logistic(0, 1.0, self.mvst.sigma.shape)

    def energy(self):
        return -self.mvst._log_prob(self.data, **self.state).sum()
