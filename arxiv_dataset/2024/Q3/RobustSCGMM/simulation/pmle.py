import warnings
import numpy as np
from scipy import linalg
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky, _compute_log_det_cholesky


def _estimate_gaussian_covariances_full(resp, X, nk, means, cov_reg, Sx):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    Sx: covariance matrix of X

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    # for k in range(n_components):
    #     diff = X - means[k]
    #     covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
    #     covariances[k].flat[::n_features + 1] += reg_covar
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) + 2 * cov_reg * Sx
        covariances[k] /= nk[k] + 2 * cov_reg
    return covariances


def _estimate_gaussian_parameters(X, resp, cov_reg, covariance_type, Sx):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    cov_reg : float
        The strength of penalty term on the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    # nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    nk = resp.sum(axis=0)
    # np.finfo : Machine limits for floating point types.
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    if covariance_type == 'full':
        covariances = _estimate_gaussian_covariances_full(
            resp, X, nk, means, cov_reg, Sx)
    return nk / nk.sum(), means, covariances


class pMLEGMM(GaussianMixture):
    """
    Extends sklearn.mixture.GaussianMixture to fit mixture via pMLE.

    Parameters
    ----------
        See sklearn.mixture.GaussianMixture
    """

    def __init__(self,
                 n_components=1,
                 covariance_type='full',
                 tol=1e-3,
                 max_iter=100,
                 n_init=1,
                 init_params='kmeans',
                 weights_init=None,
                 means_init=None,
                 precisions_init=None,
                 cov_reg=0,
                 random_state=None,
                 warm_start=False,
                 verbose=0,
                 verbose_interval=10):
        if cov_reg < 0.:
            raise ValueError("Invalid value for 'cov_reg': %.5f "
                             "regularization on covariance must be "
                             "non-negative" % cov_reg)
        self.cov_reg = cov_reg

        super().__init__(n_components=n_components,
                         tol=tol,
                         covariance_type=covariance_type,
                         reg_covar=0,
                         max_iter=max_iter,
                         n_init=n_init,
                         init_params=init_params,
                         weights_init=weights_init,
                         means_init=means_init,
                         precisions_init=precisions_init,
                         random_state=random_state,
                         warm_start=warm_start,
                         verbose=verbose,
                         verbose_interval=verbose_interval)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        self.Sx = np.cov(X.T)
        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.cov_reg, self.covariance_type, self.Sx)

        self.weights_ = (weights
                         if self.weights_init is None else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array([
                linalg.cholesky(prec_init, lower=True)
                for prec_init in self.precisions_init
            ])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def cov_penalty(self, X):
        """
        Compute the penalty term on covariance

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        # array-like, shape (n_components,)
        # The determinant of the precision cholesky matrix for each component.
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_,
                                            self.covariance_type, n_features)

        traces = np.empty((self.n_components, ))
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                traces[k] = np.trace(
                    np.dot(self.Sx, self.precisions_cholesky_[k]).dot(
                        self.precisions_cholesky_[k].T))
        else:
            for k in range(self.n_components):
                traces[k] = 0

        return -self.cov_reg * (np.sum(traces) - 2 * np.sum(log_det))

    def _compute_lower_bound(self, _, log_prob_norm, X):
        return log_prob_norm + self.cov_penalty(X) / X.shape[0]

    def _fim(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        dldmeans = np.empty((n_samples, self.n_components, n_features))
        dldcovs = np.empty(
            (n_samples, self.n_components, n_features, n_features))

        # gradient w.r.t weight
        log_resp = self._estimate_log_prob(X)
        weighted_log_prob = log_resp + np.log(self.weights_)
        dldweights = np.exp(
            (log_resp.T - logsumexp(weighted_log_prob, axis=1)).T)

        # gradient w.r.t mean & the cholesky decomposition of covariance
        score_weights = np.exp(
            (weighted_log_prob.T - logsumexp(weighted_log_prob, axis=1)).T)
        for k, (mu, prec) in enumerate(zip(self.means_, self.precisions_)):
            # gradient w.r.t mean
            sigma_inv_mean = (X - mu).dot(prec)
            dldmeans[:,
                     k, :] = score_weights[:, k][:,
                                                 np.newaxis] * sigma_inv_mean

            # gradient w.r.t cov
            prec_inv_mean = (X - mu).dot(prec)
            sigma_inv_cov_diff = np.einsum(
                'ik,ij->ikj', prec_inv_mean,
                prec_inv_mean) - prec[np.newaxis, :, :]
            dldcovs[:,
                    k, :, :] = score_weights[:,
                                             k][:, np.newaxis, np.
                                                newaxis] * sigma_inv_cov_diff / 2

        # Create a lower triangular matrix mask with the diagonal excluded
        lower_triangular_mask = np.tri(n_features, k=-1, dtype=bool)
        # Create an upper triangular matrix mask with the diagonal excluded
        upper_triangular_mask = ~np.tri(n_features, k=0, dtype=bool)
        # Multiply the lower off-diagonal elements by 2 for each 2D slice
        dldcovs[..., lower_triangular_mask] *= 2
        # Multiply the upper off-diagonal elements by 0 for each 2D slice
        dldcovs[..., upper_triangular_mask] = 0
        score = np.hstack(
            (dldweights, dldmeans.reshape(-1, self.n_components * n_features),
             dldcovs.reshape(-1, self.n_components * n_features**2)))
        fim = np.einsum('ik,ij->ikj', score, score).mean(0)

        self.fim_ = fim

    def ploglik(self, X):
        """penalized Loglikelihood of the fitted model

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        loglik: float
            The larger the better which goes to infinity under GMM.
        """
        return super().score(X) * X.shape[0] + self.cov_penalty(X)

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.cov_reg, self.covariance_type, self.Sx)
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.
        .. versionadded:: 0.20
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X,
                                dtype=[np.float64, np.float32],
                                ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError("Expected n_samples >= n_components "
                             f"but got n_components = {self.n_components}, "
                             f"n_samples = {X.shape[0]}")
        # self._check_initial_parameters(X)
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        # n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(
                    log_resp, log_prob_norm, X)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)
                if change < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.warm_start and not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        # compute the fisher information matrix
        # self._fim(X)

        return log_resp.argmax(axis=1)
