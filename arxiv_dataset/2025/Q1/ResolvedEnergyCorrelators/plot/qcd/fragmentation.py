import numpy as np

from scipy.linalg import expm

from qcd.qcd_basics import alpha_s, beta_0
import qcd.splitting_functions as qcd_sf


class DGLAPEvolver:
    def __init__(self, x_edges, x_centers,
                 splitting_functions,
                 epsilon=1e-5):
        """
        Initialize the DGLAP Evolver.

        Parameters:
        - x_edges: numpy array of x edges (momentum fraction)
        - x_centers: numpy array of x values (momentum fraction)
        - splitting_functions: set of splitting functions
        """
        # bin edges and centers
        self.x_edges = x_edges
        self.x_centers = x_centers
        self.nbins = len(x_centers)
        self.dx = np.diff(x_edges)

        if splitting_functions == 'qcd':
             self.splitting_functions = np.array([
                         [qcd_sf.P_qq, qcd_sf.P_qg],
                         [qcd_sf.P_gq, qcd_sf.P_gg]
                     ])
        else:
            self.splitting_functions = np.array(splitting_functions)
        self.num_partonic_flavors = self.splitting_functions.shape[0]

        # Regularization parameters
        self.epsilon = epsilon

        # Precompute splitting function matrices
        self._compute_splitting_matrices()

        self.evolved_distributions = None


    def _compute_splitting_matrices(self):
        self.P_matrices = []

        X = self.x_centers[:, np.newaxis]  # Shape (nbins, 1)
        Z = self.x_centers[np.newaxis, :]  # Shape (1, nbins)

        X = np.broadcast_to(X, (self.nbins, self.nbins))
        Z = np.broadcast_to(Z, (self.nbins, self.nbins))

        xi = X / Z  # Shape (nbins, nbins)

        # Mask where z >= x and xi is within a valid range
        # =======================================================
        # DEBUG/TOGGLE: No regularization
        # =======================================================
        # mask = (Z >= X) & (xi >= self.x_centers[0]) \
        #                 & (xi >= self.epsilon) \
        #                 & (xi <= 1.0 - self.epsilon)
        # =======================================================
        # DEBUG: Allowing delta-/plus-regularization
        # =======================================================
        mask = (Z >= X) & (xi >= 0) & (xi <= 1)

        # =======================================================
        # DEBUG: Something a bit weird... xi is not monotonic
        # print(f'{xi[mask]=}')
        print(f'Min xi: {xi[mask].min()}, Max xi: {xi[mask].max()}')
        print(f'{xi[mask].shape=}')
        print(f'{mask.shape=}')
        # DEBUG: Makes me think that the xis I'm putting into P_ij
        # DEBUG:   are not the same as the self.xs appearing later
        # DEBUG:   in the Mellin convolution which is the heart here
        # =======================================================

        for i in range(self.num_partonic_flavors):
            P_row = []
            for j in range(self.num_partonic_flavors):
                P_ij = np.zeros((self.nbins, self.nbins))

                P_func = self.splitting_functions[i][j]
                P_val = P_func(xi[mask], epsilon=self.epsilon)
                P_ij[mask] = P_val / Z[mask]

                # ============================================
                # DEBUG: checking for nans
                if np.isnan(P_ij).any():
                    raise ValueError("found P_ij nan")
                # ============================================

                P_row.append(P_ij)
            self.P_matrices.append(P_row)
        self.P_matrices = np.asarray(self.P_matrices)


    def evolve(self, initial_distributions, mu_i, mu_f,
               n_steps=1e4, fixed_coupling=False):
        Q_grid = np.logspace(np.log10(mu_i), np.log10(mu_f),
                             int(n_steps)+1)
        dlnQ = np.diff(np.log(Q_grid))

        evolved_distributions = np.copy(initial_distributions)

        for k in range(len(Q_grid) - 1):
            Q = Q_grid[k]
            if fixed_coupling:
                alpha_s_Q = alpha_s(mu_i)
            else:
                alpha_s_Q = alpha_s(Q)

            conv = np.zeros_like(evolved_distributions)

            for i in range(self.num_partonic_flavors):
                for j in range(self.num_partonic_flavors):
                    # P_ij = P_ij(x/z') / z'
                    P_ij = self.P_matrices[i][j]
                    # ============================================
                    # DEBUG: checking for nans
                    if np.isnan(P_ij).any():
                        raise ValueError("found P_ij nan")
                    # ============================================

                    # f(z')
                    f_j_dx = evolved_distributions[j, :] * self.dx
                    # ============================================
                    # DEBUG: checking for nans
                    if np.isnan(f_j_dx).any():
                        raise ValueError("found f_j_dx nan")
                    # DEBUG: nans found here, somehow not in P_ij
                    # ============================================

                    # Integrating dz'
                    conv_ij = P_ij.dot(f_j_dx)
                    conv[i, :] += conv_ij

            factor = -(alpha_s_Q/np.pi) * dlnQ[k]
            evolved_distributions += factor * conv

        # Storing final distribtuions
        self.evolved_distributions = evolved_distributions

        # Checking for momentum conservation
        momentum = np.sum(self.mellin_moment(2))
        if not np.isclose(1, momentum):
            raise RuntimeError("Momentum conservation violated:\n\t"
                               f"sum_j <z_j> = {momentum}.")

        return evolved_distributions


    def mellin_moment(self, N):
        return np.sum(self.evolved_distributions *
                      self.x_centers**(N-1) * self.dx, axis=1)


class QCDJetFragmenter(DGLAPEvolver):
    def __init__(self, initial_flavor, Q, R_jet, epsilon=1e-5):
        # Jet parameters
        self.initial_flavor = initial_flavor
        self.Q = Q
        self.R_jet = R_jet

        # Define the ranges for logarithmic and linear spacing
        x_min = epsilon/2
        x_mid = 0.1
        x_max = 1.0+epsilon/2

        # Number of points in each region
        N_log = 250  # Number of logarithmic points
        N_lin = 250  # Number of linear points

        # =======================================================
        # DEBUG: Adding zero-bin region for regularization/sum rule
        # =======================================================

        # Create logarithmically spaced points from 0 to x_min
        x_lin1 = np.linspace(0, x_min, N_lin, endpoint=False)
        # Create logarithmically spaced points from x_min to x_mid
        x_log = np.logspace(np.log10(x_min), np.log10(x_mid),
                            N_log, endpoint=False)
        # Create linearly spaced points from x_mid to x_max
        x_lin2 = np.linspace(x_mid, x_max, N_lin)

        # Combine the grids
        x_edges = np.concatenate([x_lin1, x_log, x_lin2])
        x_centers = np.concatenate(
                        [(x_lin1[1:]+x_lin1[:-1])/2,
                        [np.sqrt(x_lin1[-1]*x_log[0])],
                        np.sqrt(x_log[1:]*x_log[:-1]),
                        [np.sqrt(x_log[-1]*x_lin2[0])],
                        (x_lin2[1:]+x_lin2[:-1])/2]
                    )

        super().__init__(x_edges, x_centers,
                         splitting_functions='qcd',
                         epsilon=epsilon)


    def evolve(self, r_sub, **kwargs):
        mu_i, mu_f = self.Q*self.R_jet, self.Q*r_sub

        # The initial distribution is a flavor/momentum-fraction
        # delta function
        initial = np.zeros((self.num_partonic_flavors,
                            len(self.x_centers)))
        initial[self.initial_flavor, -1] = 1.0 / self.dx[-1]

        return super().evolve(initial, mu_i, mu_f, **kwargs)


def p2p_fragmentation_moment(N, mu_i, mu_f, **kwargs):
    if hasattr(N, '__len__'):
        return np.asarray([p2p_fragmentation_moment(n, mu_i, mu_f,
                                                    **kwargs)
                           for n in N])

    delta = np.log(alpha_s(mu_f, **kwargs)/alpha_s(mu_i, **kwargs))\
              /(2*np.pi*beta_0)
    return expm(delta * qcd_sf.Phat_matrix(N))
