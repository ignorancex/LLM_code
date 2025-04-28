import numpy as np
from scipy.interpolate import RegularGridInterpolator

from multiprocessing import Pool
from tqdm import tqdm

from histogram import HistogramData1D, save_histogram_to_file
from plotter import ewoc_data_dir

from qcd.qcd_basics import alpha_s
from qcd.splitting_functions import P_qq, P_qg, P_gq, P_gg
from qcd.fragmentation import QCDJetFragmenter


class MonteCarloEWOC:
    def __init__(self, initial_flavor, E_cm, R_jet, r_sub,
                 obs_bins, obs_centers,
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
        """
        self.initial_flavor = initial_flavor
        self.Q, self.R_jet, self.r_sub = E_cm/2., R_jet, r_sub
        self.epsilon = kwargs.pop('epsilon', 1e-10)
        self.obs_bins = obs_bins
        self.obs_centers = obs_centers
        self.num_samples = None

        # Allowing different levels of accuracy
        self.accuracy = kwargs.pop('accuracy', 'LL').upper()

        # Precompute a grid of theta values for evolution
        self.theta_grid = np.logspace(
                            np.log10(self.theta_min()),
                            np.log10(self.R_jet),
                            num=kwargs.pop('num_thetas', 100))

        # Precompute evolved distributions F_j_i for each theta
        self.evolver_i = QCDJetFragmenter(initial_flavor,
                                          self.Q, self.R_jet)
        self.xs = self.evolver_i.x_centers
        self.F_j_i_theta = []
        for theta in self.theta_grid:
            F_j_i = self.evolver_i.evolve(theta)
            self.F_j_i_theta.append(F_j_i)

        # Similarly, precompute F_l_k_y
        self.F_l_k_theta = []
        if self.accuracy not in ['LO', 'PRE-LL']:
            for theta in self.theta_grid:
                # Quark evolution
                evolver_q = QCDJetFragmenter(0, self.Q, theta)
                F_l_q = evolver_q.evolve(self.r_sub)
                # Gluon evolution
                evolver_g = QCDJetFragmenter(1, self.Q, theta)
                F_l_g = evolver_q.evolve(self.r_sub)

                self.F_l_k_theta.append(
                    np.transpose(np.array([F_l_q, F_l_g])))

        # Allowing additional metadata
        self.observable = kwargs.pop('observable', 'Observable')

        if kwargs:
            raise ValueError("Received unexpected keywords "
                             f"{kwargs.keys()}.")


    def obs_function(self, x, z, theta, y , yp):
        raise NotImplementedError("obs_function must be implemented "
                                  "by subclasses.")

    def alpha(self, x):
        if self.accuracy == 'MLL':
            # Freezing coupling to be <= 1
            return np.minimum(np.ones(len(x)),
                              abs(alpha_s(self.Q * x)))
        else:
            return alpha_s(self.Q * self.R_jet)


    def interpolate_F_j_i_vectorized(self, theta, x):
        """
        Interpolate F_j_i for given theta and x arrays.
        Returns an array of shape (len(x), 2), where:
        - [:, 0] corresponds to the quark distribution.
        - [:, 1] corresponds to the gluon distribution.
        """
        # Ensure that F_j_i_theta is a NumPy array of correct shape
        if not isinstance(self.F_j_i_theta, np.ndarray):
            self.F_j_i_theta = np.array(self.F_j_i_theta)
            # Shape: (num_thetas, num_flavors, num_xs)

        # Transpose F_j_i_theta to shape (thetas, xs, flavors)
        # This rearranges the axes to match the expected shape
        F_j_i_theta_transposed = np.transpose(
            self.F_j_i_theta, axes=(0, 2, 1))

        # Create the interpolator with multi-dimensional outputs
        interpolator = RegularGridInterpolator(
            (self.theta_grid, self.xs),
            F_j_i_theta_transposed,  # Shape: (ths, xs, flavors)
            bounds_error=False,
            fill_value=0
        )

        # Prepare points for interpolation
        points = np.column_stack((theta, x))  # Shape: (xs, flavors)

        # Perform interpolation
        F_j_i_values = interpolator(points)  # Shape: (xs, flavors)

        return F_j_i_values  # Shape: (len(x), 2)


    def interpolate_F_l_k_vectorized(self, theta, y):
        """
        Interpolate F_k_l for given theta and x arrays.
        Returns an array of shape (len(x), 2), where:
        - [:, 0] corresponds to the quark distribution.
        - [:, 1] corresponds to the gluon distribution.
        """
        # Shape: (thetas, flavors, flavors, ys)
        if not isinstance(self.F_l_k_theta, np.ndarray):
            self.F_l_k_theta = np.array(self.F_l_k_theta)

        # Create the interpolator with multi-dimensional outputs
        interpolator = RegularGridInterpolator(
            (self.theta_grid, self.xs),
            self.F_l_k_theta,
            bounds_error=False,
            fill_value=0
        )

        # Prepare points for interpolation
        points = np.column_stack((theta, y))
        # Shape: (ys, flavors, flavors)

        # Perform interpolation
        F_l_k_values = interpolator(points)

        return F_l_k_values  # Shape: (ys, flavors, flavors)


    def theta_min(self):
        if self.r_sub > 0:
            return self.r_sub
        return self.epsilon


    def generate_samples(self, num_samples):
        # Sample variables in batches
        samples = {}

        # Sample x, y, yp based on accuracy
        if self.accuracy == 'LO':
            samples['x'] = np.ones(num_samples)
            samples['y'] = np.ones(num_samples)
            samples['yp'] = np.ones(num_samples)
            samples['x_jacobian'] = np.ones(num_samples)
            samples['y_jacobian'] = np.ones(num_samples)
            samples['yp_jacobian'] = np.ones(num_samples)
        elif self.accuracy == 'PRE-LL' or self.r_sub == 0:
            # Sample x
            samples['x'] = self.sample_energy_fractions(num_samples)
            samples['x_jacobian'] = np.ones(num_samples)
            # y and yp are set to 1
            samples['y'] = np.ones(num_samples)
            samples['yp'] = np.ones(num_samples)
            samples['y_jacobian'] = np.ones(num_samples)
            samples['yp_jacobian'] = np.ones(num_samples)
        elif self.accuracy == 'POST-LL':
            # Set x to 1
            samples['x'] = np.ones(num_samples)
            samples['x_jacobian'] = np.ones(num_samples)
            # Sample y and yp uniformly
            samples['y'] = self.sample_energy_fractions(num_samples)
            samples['yp'] = self.sample_energy_fractions(num_samples)
            samples['y_jacobian'] = np.ones(num_samples)
            samples['yp_jacobian'] = np.ones(num_samples)
        else:
            # Sample x uniformly
            samples['x'] = self.sample_energy_fractions(num_samples)
            samples['x_jacobian'] = np.ones(num_samples)
            # Sample y and yp uniformly
            samples['y'] = self.sample_energy_fractions(num_samples)
            samples['yp'] = self.sample_energy_fractions(num_samples)
            samples['y_jacobian'] = np.ones(num_samples)
            samples['yp_jacobian'] = np.ones(num_samples)

        # Sample theta
        ln_theta_min = np.log(self.theta_min())
        ln_theta_max = np.log(self.R_jet)
        ln_theta = np.random.uniform(ln_theta_min, ln_theta_max,
                                     num_samples)
        samples['theta'] = np.exp(ln_theta)
        samples['theta_jacobian'] = samples['theta']

        # Sample z
        ln_z_min = np.log(self.epsilon)
        ln_z_max = 0
        ln_z = np.random.uniform(ln_z_min, ln_z_max,
                                 num_samples)
        samples['z'] = np.exp(ln_z)
        samples['z_jacobian'] = samples['z']

        return samples

    def sample_energy_fractions(self, num_samples):
        if self.accuracy == 'MLL':
            x_min = .14/(self.Q*self.R_jet)
            x_max = 1
        else:
            x_min, x_max = 0, 1

        return np.random.uniform(x_min, x_max, num_samples)


    def integrate(self, num_samples, batch_size=100000):
        # Initialize the histogram
        hist = np.zeros(len(self.obs_bins) - 1)
        self.num_samples = num_samples

        # Initialize number of batches and finalized results
        num_batches = int(num_samples // batch_size)
        results = []
        remaining_samples = num_samples % batch_size

        num_async = num_batches
        if remaining_samples > 0:
            num_async += 1

        # Create a pool of workers
        with Pool() as pool, tqdm(total=num_async) as pbar:
            # Deal with batches of the desired size
            for _ in range(num_batches):
                results.append(pool.apply_async(
                    self.process_batch, (batch_size,),
                    callback=lambda _: pbar.update(1)))
            # Handle remaining samples
            if remaining_samples > 0:
                results.append(pool.apply_async(
                    self.process_batch, (remaining_samples,),
                    callback=lambda _: pbar.update(1)))

            # Collect results
            for res in results:
                hist += res.get()

        # Normalize the histogram
        hist *= self.integration_area() / num_samples
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


    def integration_area(self):
        # Factors from splitting function integration
        area = np.log(1/self.epsilon)
        area *= np.log(self.R_jet/self.theta_min())

        # If MLL, energy fractions generated differently
        if self.accuracy == 'MLL':
            x_min = .14/(self.Q*self.R_jet)
            area *= (1 - x_min)**3.

        return area


    def process_batch(self, batch_size):
        samples = self.generate_samples(batch_size)
        weights, observables = self.compute_batch(samples)
        return self.accumulate_histogram(weights, observables)


    def compute_batch(self, samples):
        x = samples['x']
        z = samples['z']
        theta = samples['theta']
        y = samples['y']
        yp = samples['yp']
        jacobian = (samples['theta_jacobian'] *
                    samples['z_jacobian'] *
                    samples['x_jacobian'] *
                    samples['y_jacobian'] *
                    samples['yp_jacobian'])

        # Compute observables
        obs = self.obs_function(x, z, theta, y, yp)

        # Compute weights
        weight = self.compute_weight_vectorized(x, z, theta, y, yp)\
                    * jacobian

        return weight, obs


    def accumulate_histogram(self, weights, observables):
        hist, _ = np.histogram(observables, bins=self.obs_bins,
                               weights=weights)
        return hist


    def compute_weight_vectorized(self, x, z, theta, y, yp):
        # Vectorized version of compute_weight
        weight = np.zeros_like(x)

        # Evaluate P_{k, k' <- j}(z)
        P_k_kp_j_z = self.evaluate_splitting_function_vectorized(z)

        if self.accuracy == 'LO':
            weight = (z * (1 - z) *
                      (self.alpha(theta*z) / (theta*np.pi)) *
                      np.sum(P_k_kp_j_z[:, :, :, self.initial_flavor],
                             axis=(1, 2)))
            return weight

        # Evaluate F_j_i_x
        F_j_i_x = self.interpolate_F_j_i_vectorized(theta, x)

        if self.accuracy == 'PRE-LL' or self.r_sub == 0:
            weight = (x**2 * z*(1-z) *
                      (self.alpha(theta*x*z) / (theta*np.pi)) *
                      np.einsum('aj,akpj->a', F_j_i_x, P_k_kp_j_z))
            return weight

        # Evaluate F_l_k_y and F_lp_kp_yp
        F_l_k_y = self.interpolate_F_l_k_vectorized(theta, y)
        F_lp_kp_yp = self.interpolate_F_l_k_vectorized(theta, yp)

        # Compute the weight
        weight = (x**2 * z*(1-z) * y * yp *
                  (self.alpha(theta*x*z) / (theta*np.pi)) *
                  np.einsum('aj,akpj,alk,aqp->a',
                            F_j_i_x, P_k_kp_j_z, F_l_k_y, F_lp_kp_yp))
        return weight


    def evaluate_splitting_function_vectorized(self, z):
        """
        Evaluate the splitting functions P_{k, k' <- j}(z) for vector z.
        Returns an array P_k_kp_z of shape (len(z), num_final_flavors).
        """
        # Number of samples
        n = len(z)

        # Initialize the splitting function array
        # Two initial flavors and two final flavors
        P_k_kp_z = np.zeros((n, 2, 2, 2))

        # Initial quark
        P_k_kp_z[:, 0, 1, 0] = P_qq(z)   # q -> q + g
        P_k_kp_z[:, 1, 0, 0] = P_gq(z)   # q -> g + q

        # Initial gluon
        P_k_kp_z[:, 1, 1, 1] = P_gg(z)   # g -> g + g
        P_k_kp_z[:, 0, 0, 1] = P_qg(z)   # g -> q + qbar

        return P_k_kp_z


    def save(self, file_prefix=''):
        filename = f'{self.observable}_ewoc_{self.accuracy.lower()}_'
        if file_prefix:
          filename += file_prefix+'_'
        filename += f'jet{self.R_jet:.2f}'.replace('.', '-')+\
            f'_subjet{self.r_sub:.2f}'.replace('.', '-')+\
            f'_{self.num_samples:0.0}.py'.replace('+0', '').\
                                            replace('+', '')
        save_histogram_to_file(self.hist_data, ewoc_data_dir/filename)


class MonteCarloEEC(MonteCarloEWOC):
    def __init__(self, **kwargs):
        super().__init__(observable='theta', **kwargs)

    def obs_function(self, x, z, theta, y, yp):
        return theta


class MonteCarloMassEWOC(MonteCarloEWOC):
    def __init__(self, **kwargs):
        super().__init__(observable='mass', **kwargs)

    def obs_function(self, x, z, theta, y, yp):
        return self.Q * x * theta * np.sqrt(z*(1-z) * y*yp)
