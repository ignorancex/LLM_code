import numpy as np

from tqdm import tqdm

from histogram import HistogramData1D, save_histogram_to_file
from plotter import ewoc_data_dir

from qcd.qcd_basics import alpha_s
from qcd.splitting_functions import P_qq, P_qg, P_gq, P_gg
from qcd.fragmentation import QCDJetFragmenter


class MonteCarloEWOC:
    def __init__(self, initial_flavor, E_cm, R_jet, r_sub,
                 obs_bins, obs_centers, obs_function,
                 **kwargs):
        """
        Initialize the MonteCarloEWOC.

        Parameters:
        - initial_flavor: int, the initial parton flavor index (e.g., 0 for quark)
        - Q:
            float
            The initial scale (jet pT or energy) in GeV
        - R_jet (r_sub):
            float
            The initial (sub)jet radius
        - obs_bins/centers:
            array_like
            The bin edges/centers for the EWOC histogram
        - obs_function:
            callable
            The function f(x, z, theta, y, y') defining obs
            on a fragmentation-split-fragmentation representation
            of a pairwise observable
        """
        self.initial_flavor = initial_flavor
        self.Q, self.R_jet, self.r_sub = E_cm/2., R_jet, r_sub
        self.obs_bins = obs_bins
        self.obs_centers = obs_centers
        self.obs_function = obs_function

        # Store x grid and dx for sampling
        fragmenter = QCDJetFragmenter(0, 1000, 1)
        self.xs = fragmenter.x_centers

        # Allowing modified leading log
        self.accuracy = kwargs.pop('accuracy', 'LL').upper()
        if self.accuracy == 'MLL':
            self.alpha = lambda x: alpha_s(self.Q*x)
        else:
            self.alpha = lambda x: alpha_s(self.Q*R_jet)

        # Allowing additional metadata
        self.observable = kwargs.pop('observable', 'Observable')

        if kwargs:
            raise ValueError("Received unexpected keywords "
                             f"{kwargs.keys()}.")


    def integrate(self, num_samples=10000, epsilon=1e-7):
        """
        Perform the Monte Carlo integration and return an EWOC histogram.
        """
        # Initialize the histogram
        hist = np.zeros(len(self.obs_bins) - 1)

        # Area of integrated phase space, for Monte Carlo integration
        area = np.log(1/epsilon)
        if self.r_sub > 0:
            area *= np.log(self.R_jet/self.r_sub)
        else:
            area *= (np.log(self.R_jet) - np.log(.14/self.Q))
        if self.accuracy != 'LO':
            area *= np.log(1/epsilon)

        # Sampling loop
        for _ in tqdm(range(num_samples)):
            # -----------------------
            # Partonic splitting
            # -----------------------
            # Sample splitting angle logarithmically
            vars_jacs = [self.sample_theta()]
            # Sample momentum fraction z of first parton
            vars_jacs.append(self.sample_z(epsilon))

            # -----------------------
            # Fragmentation
            # -----------------------
            if self.accuracy.upper() == 'LO':
                # No fragmentation at LO
                for _ in ['x', 'y', 'yp']:
                    vars_jacs.append([1, 1])
            elif self.accuracy.upper() in ['PRE-LL', 'PLL']\
                    or self.r_sub == 0:
                # For a subjet radius of zero, QCD sum rules help out
                # First stage of fragmentation: Sample x
                vars_jacs.append(self.sample_z(epsilon))
                # Second stage of fragmentation: use sum rules
                for _ in ['y', 'yp']:
                    vars_jacs.append([1, 1])
            else:
                # Beyond LO, need momentum fractions from fragmentation
                # First stage of fragmentation: Sample x
                vars_jacs.append(self.sample_z(epsilon))
                # Second stage fragmentations: Sample y and y'
                vars_jacs.append(self.sample_y())
                vars_jacs.append(self.sample_y())

            theta, z, x, y, yp = [i[0] for i in vars_jacs]
            jacobian = np.prod([i[1] for i in vars_jacs])

            # -----------------------
            # Calculation
            # -----------------------
            obs = self.obs_function(x, z, theta, y, yp)
            weight = self.compute_weight(x, z, theta, y, yp)*jacobian

            # Accumulate weight into the correct histogram bin
            bin_index = np.searchsorted(self.obs_bins, obs) - 1
            if 0 <= bin_index < len(hist):
                hist[bin_index] += weight

        # Normalize the histogram
        hist *= area/num_samples
        hist /= np.diff(self.obs_bins)

        # Create HistogramData1D object
        self.hist_data = HistogramData1D(hist=hist,
              edges={self.observable: self.obs_bins},
              centers={self.observable: self.obs_centers},
              metadata={'Q': self.Q,
                        'initial_flavor': self.initial_flavor,
                        'jet_rad': self.R_jet, 'sub_rad': self.r_sub,
                        'accuracy': self.accuracy})

        return self.hist_data


    def sample_theta(self):
        """
        Sample theta from the measure dtheta / theta between
        theta_min and theta_max.
        """
        # The measure is dtheta / theta -> we sample uniformly in ln(theta)
        if self.r_sub > 0:
            ln_theta_min = np.log(self.r_sub)
        else:
            ln_theta_min = np.log(2*.14/self.Q)
        ln_theta_max = np.log(self.R_jet)
        ln_theta = np.random.uniform(ln_theta_min, ln_theta_max)
        theta = np.exp(ln_theta)
        return [theta, theta]


    def sample_z(self, epsilon):
        """
        Sample z from the splitting function P_{k, k' <- j}(z).
        For simplicity, we'll assume P(z) is known and can be sampled.
        """
        # Example: Sample z uniformly between z_min and z_max
        ln_z_min = np.log(epsilon)
        ln_z = np.random.uniform(ln_z_min, 0)
        z = np.exp(ln_z)
        return [z, z]


    def sample_y(self, epsilon=1e-5):
        """
        Sample y from F_{\ell <- k}(y; r <- theta).
        """
        # For simplicity, assume y is sampled uniformly in [0,1]
        y = np.random.uniform(0, 1)
        return [y, 1]


    def compute_weight(self, x, z, theta, y, yp):
        """
        Compute the weight for the sample point.

        Additional factors of z and theta from jacobians of logarithmic
        sampling in z and theta.
        """
        # Evaluate P_{k, k' <- j}(z)
        P_k_kp_j_z = self.evaluate_splitting_function(z)

        if self.accuracy == 'LO':
            weight = \
                (z*(1-z) * (2*self.alpha(theta*z)/(theta*np.pi)) *
                 np.sum(P_k_kp_j_z[:][:][self.initial_flavor]))
            return weight

        # Evaluate F_{j <- i}(x; theta <- R)
        evolver_i = QCDJetFragmenter(0, self.Q, self.R_jet)
        F_j_i = evolver_i.evolve(theta)
        F_j_i_x = np.array([self.evaluate_distribution(self.xs,
                                                       F_j_i[flavor], x)
                            for flavor in [0, 1]])

        # If r_sub is zero, use QCD sum rules
        if self.r_sub == 0:
            weight = \
                (x**2. * z*(1-z)
                 * (self.alpha(theta*z)/(theta*np.pi))
                 * np.einsum(
                    'j,khj->',
                    F_j_i_x, P_k_kp_j_z
                   )
                )
            return weight


        # Evaluate F_{\ell <- k}(y; r <- theta)
        F_l_k_y = []
        F_lp_kp_yp = []
        for k in [0, 1]:
            evolver_k = QCDJetFragmenter(k, self.Q, theta)
            F_l_k = evolver_k.evolve(self.r_sub)
            F_l_k_y.append([self.evaluate_distribution(self.xs,
                                                       F_l_k[k], y)
                            for flavor in [0, 1]])
            F_lp_kp_yp.append([self.evaluate_distribution(self.xs,
                                                          F_l_k[k], yp)
                            for flavor in [0, 1]])

        # Compute the weight
        weight = \
            (x**2. * z*(1-z) * y*yp
             * (self.alpha(theta*z)/(theta*np.pi))
             * np.einsum(
                'j,khj,lk,mh->',
                F_j_i_x, P_k_kp_j_z, F_l_k_y, F_lp_kp_yp
               )
            )
        return weight


    def evaluate_distribution(self, x_vals, distribution, x):
        """
        Evaluate the distribution at a given x using interpolation.
        """
        # Use linear interpolation
        value = np.interp(x, x_vals, distribution)
        return value


    def evaluate_splitting_function(self, z):
        """
        Evaluate the splitting function P(z) at a given z.
        """
        # [ [qq<-q, qq<-g], [qg<-q, qg<-g]
        #   [gq<-q, gq<-g], [gg<-q, gg<-g] ]
        P_z = np.array([[[0,       P_qg(z)], [P_qq(z),       0]],
                        [[P_gq(z),       0], [0,         P_gg(z)]]])
        return P_z


    def save(self, file_prefix=''):
        filename = f'{self.observable}_{self.accuracy.lower()}ewoc_'
        if file_prefix:
          filename += file_prefix+'_'
        filename += f'jet{self.R_jet:.2f}'.replace('.', '-')+\
                    f'_subjet{self.r_sub:.2f}'.replace('.', '-')+'.py'
        save_histogram_to_file(self.hist_data, ewoc_data_dir/filename)




# TODO: Implement
class MonteCarloMassEWOC(MonteCarloEWOC):
    pass
