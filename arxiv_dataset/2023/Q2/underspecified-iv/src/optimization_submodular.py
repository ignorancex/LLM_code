import numpy as np
import jax
import plotly.graph_objects as go
from itertools import combinations
from utils import cosine_distance

# own imports
from svd_2sls import svd_2sls, combine_svd_estimation, check_component_coverage
from utils import generate_rv, generate_model



class Experiment:
    """ general Experiment Class """

    def __init__(self, key, n):
        self._key = key
        self.n = n

    def run(self, idxs):
        x = None
        y = None
        z = None
        return x, y, z


class LinearExperiment(Experiment):
    """ representation of our artificial experiment """
    def __init__(self, key, n, alpha, beta, e, M):
        super().__init__(key, n)
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.M = M

    def run(self, lam):
        """
        Run Experiment
        Parameters
        ----------
        lam: list of boolean
            indicate which instruments are used for the particular experiment run

        Returns
        -------

        """
        self._key, _ = jax.random.split(self._key, 2)

        d, p = self.alpha.shape
        ## Generate random variables
        z, u = generate_rv(self._key, d, p, self.n)

        ## Generate dataset
        x, y, z = generate_model(self.alpha, self.beta, z, u, self.e, self.M, lam)

        return x, y, z


class BaseSet:
    def __init__(self, beta, PV, PV_comp, idxs):
        self.beta = beta
        self.PV_comp = PV_comp
        self.PV = PV
        self.idxs = idxs  # indicate the indices of the alpha that has been used


class SetProposal:
    def __init__(self,
                 n_rounds,
                 budget,
                 experiment,
                 d_max=1,
                 similarity_matrix=None,
                 BaseSet=None,
                 selection_method="similarity",
                 key=jax.random.PRNGKey(2710)
                 ):

        # parameters for method
        self._d_max = d_max
        self._budget = budget
        self._experiment = experiment
        self._selection_method = selection_method  # choice between "random", "similarity"
        self._selection_trajectory = []
        self._estimation_trajectory = []
        self._n_rounds = n_rounds
        self._iter_round = 0
        self._similarity_alpha = similarity_matrix
        self._key = key
        self._estimation_function = svd_2sls

        # initialize algorithm
        if BaseSet is None:
            self.current_idxs = []
            self.current_beta = None
            self._coverage_trajectory = []
            self._beta_trajectory = []
            self._P_trajectory = []
            self._possible_idx_sets = list(np.arange(similarity_matrix.shape[0]))

        else:
            self.current_idxs = BaseSet.idxs
            self.current_beta = BaseSet.beta
            self._coverage_trajectory.append(check_component_coverage([BaseSet.PV]))
            self._beta_trajectory.append(BaseSet.beta)
            self._P_trajectory.append(BaseSet.PV)
            self._possible_idx_sets = list(np.arange(similarity_matrix.shape[0])).remove(BaseSet.idxs)

        # parameters for evaluation
        self.figure = go.Figure()
        self._error_to_beta = []
        self._error_to_nonzero_beta = []


    def _distance(self, idx_new, idx_old, type="cosine"):
        """
        compute distance based on similarity matrix
        Parameters
        ----------
        idx_new: list
            list of indices for newly selected instruments
        idx_old: list
            list of indices for already used instruments
        type: str

        Returns
        -------
        float

        """
        if type == "cosine":
            mat1 = self._similarity_alpha[idx_new, :]
            mat2 = self._similarity_alpha[idx_old, :]

        return cosine_distance(mat2, mat1)


    def _select(self, idxs_sets):
        """
        selection function for most promising idx set
        **Note**: Implementation so far are *random* and *similarity* (matrix with pairwise similarities),
        customize if needed

        Parameters
        ----------
        idxs_sets: list

        Returns
        -------
        int: returns best index
        """

        gains = np.zeros(len(idxs_sets), dtype='float64')
        jter = 0

        for idx_set in idxs_sets:
            if self._selection_method == "similarity":
                if self.current_idxs == []:
                    gain_idx = self._distance(idx_set, idx_set) / (len(idx_set) + len(idx_set) - 1) - np.log(len(
                        idx_set))
                else:
                    gain_idx = self._distance(idx_set, self.current_idxs) / (len(idx_set) + len(self.current_idxs) -
                                                                             1) - np.log(len(idx_set))

            elif self._selection_method == "random":
                self._key, subkey = jax.random.split(self._key)
                gain_idx = jax.random.uniform(subkey)

            # Update gain and sample cost
            gains[jter] = gain_idx

            # Update count
            jter += 1

        best_idx = gains.argmax()

        return idxs_sets[best_idx]


    def _cost_budget(self, idx_list, budget_method="idx_length"):
        """
        budget computation for idx set
        **Note**: Implementation so far are *idx_length* which means direct proportional to the number of
        instruments, customize here if needed

        Parameters
        ----------
        idx_list
        budget_method

        Returns
        -------

        """
        if budget_method == "idx_length":
            return len(idx_list)
        else:
            raise NotImplementedError


    def fit(self, budget_method="idx_length"):
        """
        Fit functionality, fit the model with the parameters defined in the instantiation of the class
        Parameters
        ----------
        budget_method: str
            default *idx_length*; cost of the experiment is the number of instruments

        Returns
        -------

        """
        budget = 0.0

        while (budget < self._budget) & (len(self._possible_idx_sets) > 0) & (self._iter_round < self._n_rounds):

            # make list of possible candidates
            idxs_sets = [[list(comb) for comb in combinations(self._possible_idx_sets, comb_i + 1)] for comb_i in
                         np.arange(self._d_max)]
            idxs_sets = [item for sublist in idxs_sets for item in sublist]  # flatten the whole list


            # select most promising idx set
            best_idx = self._select(idxs_sets)

            # run experiment with updated indices
            x, y, z = self._experiment.run(best_idx)

            # estimate parameters from new experiment data
            beta_est, PV, PV_comp = self._estimation_function(x, y, z, max_iv=len(best_idx))

            # update the parameters
            self.current_idxs += best_idx

            self._selection_trajectory.append(best_idx)
            self._beta_trajectory.append(beta_est)
            self._P_trajectory.append(PV)

            # update estimator
            if self.current_beta is None:
                self.current_beta = beta_est
            else:
                beta_combined = combine_svd_estimation(self._beta_trajectory, self._P_trajectory)
                self.current_beta = beta_combined

            self._estimation_trajectory.append(self.current_beta)

            # check coverage of standard basis
            cover_vec = check_component_coverage(self._P_trajectory)
            self._coverage_trajectory.append(cover_vec)

            budget += self._cost_budget(best_idx, budget_method)

            self._possible_idx_sets = [idx for idx in self._possible_idx_sets if idx not in self.current_idxs]
            self._iter_round += 1





