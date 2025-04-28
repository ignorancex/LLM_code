# bayes.py
"""Posterior Bayesian reduced-order models."""

__all__ = [
    "BayesianODE",
    "BayesianROM",
]

import os
import abc
import h5py
import numpy as np
import scipy.stats

import opinf


class _BaseBayesianModel(abc.ABC):
    """Abstract base class for Bayesian models."""

    def __init__(self, model):
        """Store the underlying model."""
        self.__model = model

    @property
    def model(self):
        """Model whose operators / parameters are defined probabilistically."""
        return self.__model

    @abc.abstractmethod
    def rvs(self) -> np.ndarray:
        """Draw a random sample from the posterior operator distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
    ) -> np.ndarray:
        """Sample from the posterior distribution and compute the
        corresponding model solution.

        Parameters
        ----------
        initial_conditions : (num_variables,) ndarray
            Initial condition to start the simulation from.
        timepoints : (K,) ndarray
            Time domain over which to solve the equations.
        """
        raise NotImplementedError

    def solution_posterior(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        ndraws: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """Get sample statistics for the posterior distribution in the ODE
        solution space.

        Parameters
        ----------
        initial_conditions : (num_variables,) ndarray
            Initial condition to start the simulation from.
        timepoints : (K,) ndarray
            Time domain over which to solve the equations.
        ndraws : int
            Number of draws to make over the posterior.
        kwargs : dict
            Other arguments for ``predict()``.

        Returns
        -------
        list of ``ndraws`` (num_variables, K) ndarrays
            Draws from the posterior distribution in the model state space.
        """
        draws = []
        num_unstables = 0
        for _ in range(ndraws):
            draw = self.predict(
                initial_conditions=initial_conditions,
                timepoints=timepoints,
                **kwargs,
            )
            if draw.shape[1] != timepoints.size:
                num_unstables += 1
                continue
            draws.append(draw)
        if num_unstables > 0:
            print(f"\n{num_unstables}/{ndraws} DRAWS UNSTABLE")

        return draws


class BayesianODE(_BaseBayesianModel):
    """Bayesian ordinary differential equations model.

    This class is used for parameter estimation in systems of ODEs.

    Parameters
    ----------
    model
        ODE model whose parameters are described by the Normal distribution
        N(mean, inv(precision)).
    mean : (num_params,) ndarray
        Mean values for the posterior of the parameters.
    precision : (num_params, num_params) ndarray
        *INVERSE* covariance matrix for the posterior of the parameters.
    alreadyinverted : bool
        If ``True``, assume ``precision`` is the covariance matrix,
        not its inverse.
    """

    def __init__(
        self,
        model,
        mean: np.ndarray,
        precision: np.ndarray,
        *,
        alreadyinverted: bool = False,
    ):
        """Store and pre-process the mean and covariance of the posterior
        distribution for the model parameters.
        """
        # Verify mean shape.
        mean = np.array(mean)
        if mean.ndim != 1:
            raise ValueError("'mean' must be one-dimensional array")
        self.__d = mean.size

        # Verify precision shape.
        precision = np.array(precision)
        if precision.shape != (self.__d, self.__d):
            raise ValueError(
                "'precision' must be (d x d) array, "
                f"d = len(mean) = {self.__d}"
            )

        # Initialize the multivariat Normal distribution for the parameters.
        cov = precision
        if not alreadyinverted:
            cov = scipy.stats.Covariance.from_precision(precision)
        self.__randomvariable = scipy.stats.multivariate_normal(mean, cov)

        # Store the model.
        if not hasattr(model, "parameters"):
            raise AttributeError("model must have a 'parameters' attribute")
        model.parameters = mean
        _BaseBayesianModel.__init__(self, model)

    @property
    def num_params(self) -> int:
        """Number of model parameters."""
        return self.__d

    @property
    def randomvariable(self) -> scipy.stats.rv_continuous:
        """Multivariate normal random variable (scipy.stats object)."""
        return self.__randomvariable

    @property
    def mean(self) -> np.ndarray:
        """Mean values for the posterior of the parameters."""
        return self.randomvariable.mean

    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix for the parameter posterior."""
        return self.randomvariable.cov

    # Random draws ------------------------------------------------------------
    def rvs(self, nonnegative: bool = False) -> np.ndarray:
        """Draw a random sample from the posterior parameter distribution.

        Parameters
        ----------
        nonnegative : bool
            If ``True`` and any components of the sample are negative,
            discard the sample and try again.

        Returns
        -------
        (nparams,) ndarray
            A single draw from the posterior model parameters distribution.
        """
        sample = np.ravel(self.randomvariable.rvs())
        if nonnegative and np.any(sample < 0):
            return self.rvs()
        return sample

    def predict(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
    ) -> np.ndarray:
        """Sample from the posterior parameter distribution and compute the
        corresponding ODE solution.

        Parameters
        ----------
        initial_conditions : (num_variables,) ndarray
            Initial condition to start the simulation from.
        timepoints : (K,) ndarray
            Time domain over which to solve the equations.

        Returns
        -------
        (num_variables, K) ndarray
            Sampled model solution.
        """
        self.model.parameters = self.rvs()
        return self.model.solve(initial_conditions, timepoints)

    # Model persistance -------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = True) -> None:
        """Save the posterior parameters.

        Parameters
        ----------
        savefile : str
            File to save data to.
        overwrite : bool
            If False and ``savefile`` exists, raise an exception.
        """
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, "w") as hf:
            hf.create_dataset("mean", data=self.mean)
            hf.create_dataset("cov", data=self.cov)

    @classmethod
    def load(cls, loadfile: str):
        """Load a previously saved Bayesian model."""
        with h5py.File(loadfile, "r") as hf:
            mean = hf["mean"][:]
            cov = hf["cov"][:]

        return cls(mean, cov, alreadyinverted=True)


class BayesianROM(_BaseBayesianModel):
    """Bayesian reduced-order model class.

    This class is used for probabilistic operator inference models.

    Parameters
    ----------
    means : list of r (d,) ndarrays
        Mean values for each row of the model operators.
    precisions : list of r (d, d) ndarrays
        *INVERSE* covariance matrices for each row of the model operators.
    model : opinf model object
        Initialized (but untrained) reduced-order model.
    alreadyinverted : bool
        If ``True``, assume ``precisions`` is the collection of covariance
        matrices, not their inverses.
    """

    def __init__(self, means, precisions, model, *, alreadyinverted=False):
        """Store and pre-process the mean and covariance of the posterior
        distribution for the model parameters.
        """
        if (r := len(means)) != (_r2 := len(precisions)):
            raise ValueError(f"len(means) = {r} != {_r2} = len(precisions)")

        self.__r = r
        self.__randomvariables = []

        for i in range(self.__r):
            # Verify dimensions.
            mean_i, cov_i = means[i], precisions[i]
            if not isinstance(mean_i, np.ndarray) or mean_i.ndim != 1:
                raise ValueError(f"means[{i}] should be 1D ndarray")
            if not isinstance(cov_i, np.ndarray) or cov_i.ndim != 2:
                raise ValueError(f"precisions[{i}] should be 2D")
            d = mean_i.shape[0]
            if cov_i.shape != (d, d):
                raise ValueError("means and precisions not aligned")
            # Make a multivariate Normal distribution for this operator row.
            if not alreadyinverted:
                cov_i = scipy.stats.Covariance.from_precision(cov_i)
            self.__randomvariables.append(
                scipy.stats.multivariate_normal(mean=mean_i, cov=cov_i)
            )

        # If operator rows are all the same size, wrap rvs() output as array.
        self.__rvsasarray = False
        d = means[0].size
        if all(mean.size == d for mean in means):
            self.__rvsasarray = True

        for attr in (
            "state_dimension",
            "_extract_operators",
            "predict",
            "ivp_method",
        ):
            if not hasattr(model, attr):
                raise AttributeError(f"model missing required member '{attr}'")
        if model.state_dimension is None:
            model.state_dimension = r
        if model.state_dimension != r:
            raise ValueError("model not aligned with distribution dimensions")
        _BaseBayesianModel.__init__(self, model)

    @property
    def ndims(self):
        """Number of reduced-order modes."""
        return self.__r

    @property
    def randomvariables(self):
        """Multivariate normal random variables (scipy.stats object)
        for each row of the operator matrix.
        """
        return self.__randomvariables

    @property
    def means(self):
        """Mean vectors for each row of the operator matrix."""
        return [rv.mean for rv in self.randomvariables]

    @property
    def covs(self):
        """Covariance matrices for each row of the operator matrix."""
        return [rv.cov for rv in self.randomvariables]

    # Random draws ------------------------------------------------------------
    def rvs(self):
        """Draw a random sample from the posterior parameter distribution."""
        ohats = [rv.rvs()[0] for rv in self.randomvariables]
        return np.array(ohats) if self.__rvsasarray else ohats

    def predict(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        input_func=None,
    ):
        """Sample from the posterior operator distribution and compute the
        corresponding ROM solution.

        Parameters
        ----------
        initial_conditions : (NUMVARS,) ndarray
            Initial condition to start the simulation from.
        timepoints : (k,) ndarray
            Time domain over which to solve the equations.
        input_func : callable
            Input function.
        """
        self.model._extract_operators(self.rvs())
        return self.model.predict(
            initial_conditions,
            timepoints,
            input_func,
            method=self.model.ivp_method,
        )

    # Model persistance -------------------------------------------------------
    def save(self, savefile, overwrite=True):
        """Save the posterior parameters.

        Parameters
        ----------
        savefile : str
            File to save data to.
        overwrite : bool
            If False and ``savefile`` exists, raise an exception.
        """
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, "w") as hf:
            hf.create_dataset("state_dimension", data=[self.ndims])
            for i, (mean_i, cov_i) in enumerate(zip(self.means, self.covs)):
                hf.create_dataset(f"means_{i}", data=mean_i)
                hf.create_dataset(f"covs_{i}", data=cov_i)
            self.model.save(hf.create_group("model"))

    @classmethod
    def load(cls, loadfile):
        """Load a previously saved Bayesian model."""
        with h5py.File(loadfile, "r") as hf:
            r = int(hf["state_dimension"][0])
            means = [hf[f"means_{i}"][:] for i in range(r)]
            covs = [hf[f"covs_{i}"][:] for i in range(r)]
            model = opinf.models.ContinuousModel.load(hf["model"])

        return cls(means, covs, model, alreadyinverted=True)
