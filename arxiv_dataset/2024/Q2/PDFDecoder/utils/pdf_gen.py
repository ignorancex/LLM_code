import numpy as np

from scipy.special import gamma
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PDFDataSet:

    def __init__(self):

        super(PDFDataSet, self).__init__()

        self.random_state = 16
        self.xs = np.array([])

        self.xtrain = np.array([])
        self.xtest = np.array([])
        self.xvalid = np.array([])

        self.ytrain = np.array([])
        self.ytest = np.array([])
        self.yvalid = np.array([])

        self.sc = StandardScaler()
        self.moment_sc = StandardScaler()
        self.scaled = 0
        self.moment_scaled = 0

        self.uplus_std = np.array([])
        self.dplus_std = np.array([])
        self.uminus_std = np.array([])
        self.dminus_std = np.array([])

        self.train_indices = np.array([])
        self.valid_indices = np.array([])
        self.test_indices = np.array([])

    def moment_calc(self, n, alpha, beta, gam, delta):

        moment = gamma(beta + 1) * ((gamma(n + alpha + 1) * (alpha + beta + delta *
                                                             (alpha + n + 1) + n + 2)) / (gamma(n + alpha + beta + 3))
                                    + (gam * gamma(n + alpha + 1.5)) / (gamma(n + alpha + beta + 2.5)))
        return moment

    def create_data(self, num_params=5, xmin=1e-5, num_pdfs=1000, num_xpoints=196,
                    latent_size=32, test_size=0.3, divx_plus=0, divx_minus=0, err_param=0.,
                    pmerr_factor=1.):

        self.xs = np.logspace(np.log10(xmin), np.log10(0.999), num_xpoints)
        self.latent_size = latent_size

        np.random.seed(self.random_state)

        alpha_dplus = np.random.uniform(-0.25 - divx_plus, 0.5 - divx_plus, size=(num_pdfs, 1))
        beta_dplus = np.random.uniform(1, 5, size=(num_pdfs, 1))
        gamma_dplus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))
        delta_dplus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))

        alpha_uplus = np.random.uniform(-0.25 - divx_plus, 0.5 - divx_plus, size=(num_pdfs, 1))
        beta_uplus = np.random.uniform(1, 5, size=(num_pdfs, 1))
        gamma_uplus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))
        delta_uplus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))

        alpha_uminus = np.random.uniform(0.5 - divx_minus, 1.25 - divx_minus, size=(num_pdfs, 1))
        beta_uminus = np.random.uniform(1, 5, size=(num_pdfs, 1))
        gamma_uminus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))
        delta_uminus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))

        alpha_dminus = np.random.uniform(0.5 - divx_minus, 1.25 - divx_minus, size=(num_pdfs, 1))
        beta_dminus = np.random.uniform(1, 5, size=(num_pdfs, 1))
        gamma_dminus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))
        delta_dminus = np.random.uniform(0.1, 9.9, size=(num_pdfs, 1))

        uplus_err = np.random.normal(loc=0, scale=err_param * pmerr_factor / ((self.xs ** alpha_uplus)
                                                                              * ((1 - self.xs) ** beta_uplus)),
                                     size=(num_pdfs, num_xpoints))
        uminus_err = np.random.normal(loc=0, scale=err_param / (
                (self.xs ** alpha_uminus) * ((1 - self.xs) ** beta_uminus)),
                                      size=(num_pdfs, num_xpoints))
        dplus_err = np.random.normal(loc=0, scale=err_param * pmerr_factor / ((self.xs ** alpha_dplus)
                                                                              * ((1 - self.xs) ** beta_dplus)),
                                     size=(num_pdfs, num_xpoints))
        dminus_err = np.random.normal(loc=0, scale=err_param / (
                (self.xs ** alpha_dminus) * ((1 - self.xs) ** beta_dminus)),
                                      size=(num_pdfs, num_xpoints))

        if num_params <= 3:
            gamma_uplus, delta_uplus = 0, 0
            gamma_uminus, delta_uminus = 0, 0
            gamma_dplus, delta_dplus = 0, 0
            gamma_dminus, delta_dminus = 0, 0

        elif num_params <= 4:
            delta_uplus = 0
            delta_uminus = 0
            delta_dplus = 0
            delta_dminus = 0

        Norm_dplus = np.random.uniform(0.25, 1.25, size=(num_pdfs, 1))
        Norm_uplus = np.random.uniform(0.25, 1.25, size=(num_pdfs, 1))

        Norm_uminus = 2. / self.moment_calc(n=-1 + divx_minus, alpha=alpha_uminus,
                                            beta=beta_uminus, gam=gamma_uminus, delta=delta_uminus)
        Norm_dminus = 1. / self.moment_calc(n=-1 + divx_minus, alpha=alpha_dminus,
                                            beta=beta_dminus, gam=gamma_dminus, delta=delta_dminus)

        u_plus_ubar = Norm_uplus * (self.xs ** alpha_uplus) * (1. - self.xs) ** beta_uplus \
                      * (1. + gamma_uplus * np.sqrt(self.xs)
                         + delta_uplus * self.xs)
        u_minus_ubar = Norm_uminus * (self.xs ** alpha_uminus) * (1. - self.xs) ** beta_uminus \
                       * (1. + gamma_uminus * np.sqrt(self.xs)
                          + delta_uminus * self.xs)
        d_plus_dbar = Norm_dplus * (self.xs ** alpha_dplus) * (1. - self.xs) ** beta_dplus \
                      * (1. + gamma_dplus * np.sqrt(self.xs)
                         + delta_dplus * self.xs)
        d_minus_dbar = Norm_dminus * (self.xs ** alpha_dminus) * (1. - self.xs) ** beta_dminus \
                       * (1. + gamma_dminus * np.sqrt(self.xs)
                          + delta_dminus * self.xs)

        self.uplus_std = u_plus_ubar * err_param * pmerr_factor / (
                (self.xs ** alpha_uplus) * ((1 - self.xs) ** beta_uplus))
        self.uminus_std = u_minus_ubar * err_param / ((self.xs ** alpha_uminus) * ((1 - self.xs) ** beta_uminus))
        self.dplus_std = d_plus_dbar * err_param * pmerr_factor / (
                (self.xs ** alpha_dplus) * ((1 - self.xs) ** beta_dplus))
        self.dminus_std = d_minus_dbar * err_param / ((self.xs ** alpha_dminus) * ((1 - self.xs) ** beta_dminus))

        u_plus_ubar = u_plus_ubar * (1. + uplus_err)
        u_minus_ubar = u_minus_ubar * (1. + uminus_err)
        d_plus_dbar = d_plus_dbar * (1. + dplus_err)
        d_minus_dbar = d_minus_dbar * (1. + dminus_err)

        pdf_data = np.hstack([d_plus_dbar, u_plus_ubar, u_minus_ubar, d_minus_dbar])

        minus_mellin_n = np.arange(-1 + divx_minus, np.floor(latent_size / 2) - 2 + divx_minus, 2)
        plus_mellin_n = np.arange(0 + divx_plus, np.floor(latent_size / 2) - 1 + divx_plus, 2)

        u_plus_ubar_moments = Norm_uplus * self.moment_calc(plus_mellin_n, alpha_uplus,
                                                            beta_uplus, gamma_uplus, delta_uplus)
        d_plus_dbar_moments = Norm_dplus * self.moment_calc(plus_mellin_n, alpha_dplus,
                                                            beta_dplus, gamma_dplus, delta_dplus)
        u_minus_ubar_moments = Norm_uminus * self.moment_calc(minus_mellin_n, alpha_uminus,
                                                              beta_uminus, gamma_uminus, delta_uminus)
        d_minus_dbar_moments = Norm_dminus * self.moment_calc(minus_mellin_n, alpha_dminus,
                                                              beta_dminus, gamma_dminus, delta_dminus)

        u_tot_moments = np.array([[i, j] for i, j in zip(u_minus_ubar_moments.ravel(),
                                                         u_plus_ubar_moments.ravel())]).reshape(num_pdfs,
                                                                                                int(latent_size / 2))

        d_tot_moments = np.array([[i, j] for i, j in zip(d_minus_dbar_moments.ravel(),
                                                         d_plus_dbar_moments.ravel())]).reshape(num_pdfs,
                                                                                                int(latent_size / 2))

        moments = np.concatenate((u_tot_moments, d_tot_moments), axis=1)

        y_indices = np.arange(0, num_pdfs, 1)

        x_train, x_valid, y_train, y_valid = train_test_split(
            pdf_data, y_indices, test_size=test_size, shuffle=True,
            random_state=self.random_state
        )
        x_valid, x_test, y_valid, y_test = train_test_split(
            x_valid, y_valid, test_size=0.5, shuffle=True, random_state=self.random_state
        )

        self.xtrain = x_train
        self.xtest = x_test
        self.xvalid = x_valid

        self.ytrain = moments[y_train]
        self.ytest = moments[y_test]
        self.yvalid = moments[y_valid]

        self.train_indices = y_train
        self.valid_indices = y_valid
        self.test_indices = y_test

    def load_data_from_file(self, file_list):

        data_set = []
        for _, filepath in enumerate(file_list):
            data = np.load(filepath)
            data_set.append(data)

        self.xtrain, self.xvalid, self.xtest, self.ytrain, self.yvalid, self.ytest = data_set

        return self.xtrain, self.xvalid, self.xtest, self.ytrain, self.yvalid, self.ytest

    def save_data_to_file(self, dir_path, string):

        np.save(dir_path / ('x_train_' + string + '.npy'), self.xtrain)
        np.save(dir_path / ('x_valid_' + string + '.npy'), self.xvalid)
        np.save(dir_path / ('x_test_' + string + '.npy'), self.xtest)

        np.save(dir_path / ('y_train_' + string + '.npy'), self.ytrain)
        np.save(dir_path / ('y_valid_' + string + '.npy'), self.yvalid)
        np.save(dir_path / ('y_test_' + string + '.npy'), self.ytest)

    def scale_data(self):

        if self.scaled == 0:
            self.xtrain = self.sc.fit_transform(self.xtrain) / 10
            self.xvalid = self.sc.transform(self.xvalid) / 10
            self.xtest = self.sc.transform(self.xtest) / 10

            self.scaled = 1

        else:
            print("Data already scaled")

    def get_pdfs(self):

        return self.xtrain, self.xvalid, self.xtest

    def scale_moments(self):

        if self.moment_scaled == 0:
            self.ytrain = self.moment_sc.fit_transform(self.ytrain)
            self.yvalid = self.moment_sc.transform(self.yvalid)
            self.ytest = self.moment_sc.transform(self.ytest)

            self.moment_scaled = 1
        else:
            print("Moments already scaled")

    def get_moments(self):

        return self.ytrain, self.yvalid, self.ytest

    def unscale_data(self):

        if self.scaled == 0:
            print("Data not scaled")

        else:
            self.xtrain = self.sc.inverse_transform(10 * self.xtrain)
            self.xvalid = self.sc.inverse_transform(10 * self.xvalid)
            self.xtest = self.sc.inverse_transform(10 * self.xtest)

            self.scaled = 0

    def unscale_moments(self):

        if self.moment_scaled == 0:
            print("Moments not scaled")

        else:
            self.ytrain = self.moment_sc.inverse_transform(self.ytrain)
            self.yvalid = self.moment_sc.inverse_transform(self.yvalid)
            self.ytest = self.moment_sc.inverse_transform(self.ytest)

            self.moment_scaled = 0
