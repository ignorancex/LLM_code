# -*- coding: utf-8 -*-
"""
Kernel Ridge Regression Causal Estimator (unified for discrete/continuous treatment).If you use this code or any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.

@author: Yordan P. Raykov, Hengrui Luo 

"""
import numpy as np
from fdasrsf import fdawarp

class KernelRidgeRegressionCausalEstimator:
    def __init__(
        self,
        lambd1,
        lambd2=None,
        lambd3=None,
        kernel_sigma=0.05,
        treatment='discrete',
        use_operator_valued_kernel=False,
        apply_srfv_Y=False,
        srfv_Y_groups=None,
        apply_srfv_X=False,
        srfv_X_groups=None
    ):
        """
        A Kernel Ridge Regression-based causal estimator that can handle
        both discrete and continuous treatments, univariate or functional outcomes,
        and optionally incorporate a second set of covariates V to compute CATE.

        Parameters
        ----------
        lambd1 : float
            Regularization parameter for the (X+D) kernel.
        lambd2 : float, optional
            Regularization parameter for the V kernel (if using V).
        lambd3 : float, optional
            Regularization parameter for the combined kernel (X+D+V) (if using V).
        kernel_sigma : float
            Bandwidth parameter (sigma) for the RBF kernel.
        treatment : {'discrete', 'continuous'}
            Indicates whether D is binary or continuous.
        use_operator_valued_kernel : bool
            Whether to use operator-valued kernels for correlated outcomes.
        apply_srfv_Y : bool
            Whether to apply SRVF registration on the outcome curves Y.
        srfv_Y_groups : list, optional
            List of treatment values for which Y will be registered.
            For 'discrete' treatment, defaults to [0,1] if None is given.
        apply_srfv_X : bool
            Whether to apply SRVF registration on the main covariates X (if shaped as curves).
        srfv_X_groups : list, optional
            List of treatment values for which X will be registered.
            For 'discrete' treatment, defaults to [0,1] if None is given.
        """
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.kernel_sigma = kernel_sigma
        self.treatment = treatment.lower()
        self.use_operator_valued_kernel = use_operator_valued_kernel

        self.apply_srfv_Y = apply_srfv_Y
        self.srfv_Y_groups = srfv_Y_groups
        self.apply_srfv_X = apply_srfv_X
        self.srfv_X_groups = srfv_X_groups

        # Decide whether we'll use V-based logic (CATE, combined kernels, etc.)
        self.use_V = (lambd2 is not None) and (lambd3 is not None)

        # If the user does not specify groups for discrete, default to [0, 1].
        if self.treatment == 'discrete':
            if self.srfv_Y_groups is None:
                self.srfv_Y_groups = [0, 1]
            if self.srfv_X_groups is None:
                self.srfv_X_groups = [0, 1]

    def _rbf_kernel(self, X1, X2):
        """
        Compute the RBF (Gaussian) kernel between X1 and X2.
        Supports 2D inputs (samples x features) or 3D (samples x timepoints x dimension).
        """
        assert X1.ndim == X2.ndim, "Input arrays must have the same number of dimensions."
        if X1.ndim == 2:
            # Standard RBF kernel for 2D inputs
            squared_norms_1 = np.sum(X1**2, axis=1, keepdims=True)
            squared_norms_2 = np.sum(X2**2, axis=1, keepdims=True)
            K = squared_norms_1 + squared_norms_2.T - 2 * np.dot(X1, X2.T)
            return np.exp(-K / (2 * self.kernel_sigma**2))
        elif X1.ndim == 3:
            # RBF for time-series or image-like data
            X1_expanded = X1[:, np.newaxis, :, :]
            X2_expanded = X2[np.newaxis, :, :, :]
            squared_diffs = (X1_expanded - X2_expanded)**2
            dists = np.sum(squared_diffs, axis=(2, 3))
            return np.exp(-dists / (2 * self.kernel_sigma**2))
        else:
            raise ValueError("Input dimensions for RBF kernel not supported.")

    def _smooth_and_register(self, data_matrix, groups, is_X=False):
        """
        Helper to smooth and apply SRVF registration to `data_matrix`.
        For discrete treatment, it registers each group in `groups` separately.
        For continuous, it registers all at once (groups is just None or all samples).

        Parameters
        ----------
        data_matrix : ndarray, shape (n_samples, n_timepoints)
            The data to be smoothed and registered in each row.
        groups : list of discrete treatment values to handle separately (if discrete),
                 or None if continuous.
        is_X : bool, indicates whether this is for X or Y (just for naming clarity).

        Returns
        -------
        data_matrix_registered : ndarray of same shape
            Registered curves.
        """
        n_samples, n_timepoints = data_matrix.shape
        # Simple smoothing
        window_size = 5
        pad_width = (window_size - 1) // 2
        data_smooth = np.zeros_like(data_matrix)
        for i in range(n_samples):
            padded = np.pad(data_matrix[i, :], pad_width, mode='edge')
            smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
            data_smooth[i, :] = smoothed[:n_timepoints]

        # If discrete, register per group. If continuous, register everything at once.
        if self.treatment == 'discrete':
            for g in groups:
                # Indices for that group
                idx = np.where(self.D_train.squeeze() == g)[0]
                if len(idx) == 0:
                    continue  # No samples for that group
                data_to_reg = data_smooth[idx, :]
                grid_T = np.linspace(0, 1, n_timepoints)

                fdasrsf_obj = fdawarp(f=data_to_reg.T, time=grid_T)
                fdasrsf_obj.srsf_align(parallel=True)
                data_registered = fdasrsf_obj.fn.T

                data_smooth[idx, :] = data_registered

            return data_smooth

        else:  # continuous
            grid_T = np.linspace(0, 1, n_timepoints)
            fdasrsf_obj = fdawarp(f=data_smooth.T, time=grid_T)
            fdasrsf_obj.srsf_align(parallel=True)
            data_registered = fdasrsf_obj.fn.T
            return data_registered

    def fit(self, X, V=None, D=None, Y=None):
        """
        Fit the estimator to the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_timepoints)
        V : ndarray, shape (n_samples, n_features_v) or (n_samples, n_timepoints_v) or None
            Additional covariates for computing CATE. If self.use_V is True,
            this must not be None.
        D : ndarray, shape (n_samples, 1)
            The treatment variable. Either discrete {0,1} or continuous in [0,1], etc.
        Y : ndarray, shape (n_samples, n_timepoints) or (n_samples,)
            Outcomes. Can be univariate or functional.
        """
        # Basic assignments
        self.X_train = X
        self.D_train = D
        self.Y_train = Y
        self.n_samples = X.shape[0]

        # If we plan to use V but it's None, raise an error
        if self.use_V:
            if V is None:
                raise ValueError(
                    "You indicated you want to use V (via lambd2,lambd3) but passed V=None."
                )
            self.V_train = V
        else:
            # If we don't plan to use V, ignore or set to None
            self.V_train = None

        # Reshape Y if univariate
        if self.Y_train.ndim == 1:
            self.Y_train = self.Y_train.reshape(-1, 1)
        self.n_timepoints = self.Y_train.shape[1]

        # 1) SRVF Registration on Y if requested
        if self.apply_srfv_Y:
            self.Y_train = self._smooth_and_register(
                data_matrix=self.Y_train,
                groups=self.srfv_Y_groups if self.treatment == 'discrete' else None,
                is_X=False
            )

        # 2) SRVF Registration on X if requested
        if self.apply_srfv_X:
            if self.X_train.ndim == 1:
                self.X_train = self.X_train.reshape(-1, 1)
            self.X_train = self._smooth_and_register(
                data_matrix=self.X_train,
                groups=self.srfv_X_groups if self.treatment == 'discrete' else None,
                is_X=True
            )

        # 3) Build the K_DD kernel based on discrete/continuous
        if self.treatment == 'discrete':
            # Dot-product kernel for D
            self.K_DD = self.D_train @ self.D_train.T
        else:
            # RBF kernel for continuous D
            self.K_DD = self._rbf_kernel(self.D_train, self.D_train)

        # 4) Build the X-related kernel matrix
        self.K_XX = self._rbf_kernel(self.X_train, self.X_train)
        self.K_XX_DD = np.multiply(self.K_XX, self.K_DD)

        # 5) If V is used, build the V kernel and combined kernel
        if self.use_V:
            self.K_VV = self._rbf_kernel(self.V_train, self.V_train)
            self.K_XX_DD_VV = np.multiply(self.K_XX_DD, self.K_VV)
        else:
            self.K_VV = None
            self.K_XX_DD_VV = None

        # 6) Compute regularized inverses for the relevant kernels
        eye_n = np.eye(self.n_samples)
        self.reg_inv = np.linalg.inv(self.K_XX_DD + self.n_samples * self.lambd1 * eye_n)

        if self.use_V:
            self.reg_inv_VV = np.linalg.inv(self.K_VV + self.n_samples * self.lambd2 * eye_n)
            self.reg_inv_DDVVXX = np.linalg.inv(self.K_XX_DD_VV + self.n_samples * self.lambd3 * eye_n)
        else:
            self.reg_inv_VV = None
            self.reg_inv_DDVVXX = None

        # 7) Operator-valued kernel logic or standard kernel logic
        if not self.use_operator_valued_kernel:
            # =========== Standard kernel for each timepoint ============= #
            self.theta_ate_vector_per_t = np.zeros((self.n_samples, self.n_timepoints))

            # We'll only compute CATE if self.use_V is True
            if self.use_V:
                self.theta_cate_vector_per_t = np.zeros((self.n_samples, self.n_samples, self.n_timepoints))
            else:
                self.theta_cate_vector_per_t = None

            for tt in range(self.n_timepoints):
                Y_tt = self.Y_train[:, tt]
                # ========== ATE ========== #
                for j in range(self.n_samples):
                    K_Dd = self.K_DD[:, j]
                    K_Xx_j = self.K_XX[:, j]
                    K_Dd_Xx_j = np.multiply(K_Dd, K_Xx_j)
                    theta_ate = Y_tt.T @ self.reg_inv @ K_Dd_Xx_j / self.n_samples
                    self.theta_ate_vector_per_t[j, tt] = theta_ate

                # ========== CATE (only if V is provided) ========== #
                if self.use_V:
                    for j in range(self.n_samples):
                        K_Dd = self.K_DD[:, j]
                        K_Xx_j = self.K_XX[:, j]
                        K_Dd_Xx_j = np.multiply(K_Dd, K_Xx_j)
                        for i in range(self.n_samples):
                            K_Vv_i = self.K_VV[:, i]
                            K_combined = np.multiply(K_Dd_Xx_j, K_Vv_i)
                            theta_cate = Y_tt.T @ self.reg_inv_DDVVXX @ K_combined
                            self.theta_cate_vector_per_t[j, i, tt] = theta_cate

        else:
            # =========== Operator-Valued Kernel ============= #
            # For correlated outcomes across timepoints
            self.K_y = np.cov(self.Y_train.T)  # shape (n_timepoints, n_timepoints)

            if not self.use_V:
                # OVK without V => only K_XX_DD
                self.K_XX_DD_Y = np.kron(self.K_XX_DD, self.K_y)
                reg_dim = self.n_samples * self.n_timepoints
                eye_big = np.eye(reg_dim)
                self.reg_inv_XX_DD_Y = np.linalg.inv(self.K_XX_DD_Y + self.n_samples * self.lambd1 * eye_big)

                Y_flat = self.Y_train.T.flatten()
                self.alpha = self.reg_inv_XX_DD_Y @ Y_flat

                # ATE
                self.theta_ate_vector_per_t = np.zeros((self.n_samples, self.n_timepoints))
                for j in range(self.n_samples):
                    K_Dd_Xx_j = self.K_XX_DD[j, :]
                    K_big = np.kron(K_Dd_Xx_j, self.K_y)
                    theta_ate = K_big @ self.alpha
                    self.theta_ate_vector_per_t[j, :] = theta_ate

                # No V => no CATE
                self.theta_cate_vector_per_t = None

            else:
                # OVK with V => use K_XX_DD_VV
                self.K_XX_DD_VV_Y = np.kron(self.K_XX_DD_VV, self.K_y)
                reg_dim = self.n_samples * self.n_timepoints
                eye_big = np.eye(reg_dim)
                self.reg_inv_XX_DD_VV_Y = np.linalg.inv(self.K_XX_DD_VV_Y + self.n_samples * self.lambd3 * eye_big)

                Y_flat = self.Y_train.T.flatten()
                self.alpha = self.reg_inv_XX_DD_VV_Y @ Y_flat

                # ATE
                self.theta_ate_vector_per_t = np.zeros((self.n_samples, self.n_timepoints))
                for j in range(self.n_samples):
                    K_Dd_Xx_j = self.K_XX_DD[j, :]
                    K_big = np.kron(K_Dd_Xx_j, self.K_y)
                    theta_ate = K_big @ self.alpha
                    self.theta_ate_vector_per_t[j, :] = theta_ate

                # CATE
                self.theta_cate_vector_per_t = np.zeros((self.n_samples, self.n_samples, self.n_timepoints))
                for j in range(self.n_samples):
                    K_Dd_j = self.K_DD[:, j]
                    K_Xx_j = self.K_XX[:, j]
                    K_Dd_Xx_j = np.multiply(K_Dd_j, K_Xx_j)
                    for i in range(self.n_samples):
                        K_Vv_i = self.K_VV[:, i]
                        K_combined = np.multiply(K_Dd_Xx_j, K_Vv_i)
                        K_big = np.kron(K_combined, self.K_y)
                        theta_cate = K_big @ self.alpha
                        self.theta_cate_vector_per_t[j, i, :] = theta_cate

        return self.theta_ate_vector_per_t, self.theta_cate_vector_per_t

    def predict(self, X_new, V_new=None, D_new=None):
        """
        Predict ATE and (optionally) CATE for new data (X_new, V_new, D_new).

        Parameters
        ----------
        X_new : ndarray of shape (n_new_samples, n_features or timepoints)
        V_new : ndarray of shape (n_new_samples, n_features_v or timepoints_v) or None
                Only used if self.use_V is True; otherwise ignored.
        D_new : ndarray of shape (n_new_samples, 1)
            New treatment values (discrete or continuous).

        Returns
        -------
        predictions_ate : ndarray of shape (n_new_samples, n_timepoints)
            Predicted ATE values at each timepoint.
        predictions_cate : ndarray of shape (n_new_samples, n_new_samples_v, n_timepoints) or None
            Predicted CATE values (if self.use_V is True and V_new is provided),
            otherwise None.
        """
        # If we apply SRVF to X_new, do a simple smoothing approach (placeholder)
        if self.apply_srfv_X:
            n_timepoints_X = X_new.shape[1] if X_new.ndim > 1 else 1
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            window_size = 5
            pad_width = (window_size - 1) // 2
            X_new_smooth = np.zeros_like(X_new)
            for i in range(X_new.shape[0]):
                padded_X = np.pad(X_new[i, :], pad_width, mode='edge')
                smoothed_X = np.convolve(padded_X, np.ones(window_size)/window_size, mode='valid')
                X_new_smooth[i, :] = smoothed_X[:n_timepoints_X]
            X_new = X_new_smooth

        # Compute K_DD_new
        if self.treatment == 'discrete':
            K_DD_new = D_new @ self.D_train.T
        else:
            K_DD_new = self._rbf_kernel(D_new, self.D_train)

        # Compute K_XX_new
        K_XX_new = self._rbf_kernel(X_new, self.X_train)
        K_Dd_Xx_new = np.multiply(K_DD_new, K_XX_new)

        # We'll always compute predictions_ate
        n_new_X = X_new.shape[0]
        predictions_ate = np.zeros((n_new_X, self.n_timepoints))

        # Decide if we can/should do CATE
        do_cate = self.use_V and (V_new is not None)

        if not self.use_operator_valued_kernel:
            # =========== Standard kernel predictions ============= #
            for tt in range(self.n_timepoints):
                Y_tt = self.Y_train[:, tt]
                # ATE
                for j in range(n_new_X):
                    K_Dd = K_DD_new[j, :]
                    K_Xx_j = K_XX_new[j, :]
                    K_Dd_Xx_j = np.multiply(K_Dd, K_Xx_j)
                    theta_ate = Y_tt.T @ self.reg_inv @ K_Dd_Xx_j / self.n_samples
                    predictions_ate[j, tt] = theta_ate

            if do_cate:
                # Build the V kernel if we have new V
                if V_new.ndim == 1:
                    V_new = V_new.reshape(-1, 1)
                K_VV_new = self._rbf_kernel(V_new, self.V_train)

                n_new_V = V_new.shape[0]
                predictions_cate = np.zeros((n_new_X, n_new_V, self.n_timepoints))
                for tt in range(self.n_timepoints):
                    Y_tt = self.Y_train[:, tt]
                    for j in range(n_new_X):
                        K_Dd = K_DD_new[j, :]
                        K_Xx_j = K_XX_new[j, :]
                        K_Dd_Xx_j = np.multiply(K_Dd, K_Xx_j)
                        for i in range(n_new_V):
                            K_Vv_i = K_VV_new[i, :]
                            K_combined = np.multiply(K_Dd_Xx_j, K_Vv_i)
                            theta_cate = Y_tt.T @ self.reg_inv_DDVVXX @ K_combined
                            predictions_cate[j, i, tt] = theta_cate
                return predictions_ate, predictions_cate
            else:
                # No V => no CATE
                return predictions_ate

        else:
            # =========== Operator-Valued Kernel predictions ============= #
            # We'll need self.alpha from training time
            # If no V is used, we had alpha for K_XX_DD
            if not self.use_V:
                # OVK but no V => K_XX_DD only
                for j in range(n_new_X):
                    K_Dd_Xx_j = K_Dd_Xx_new[j, :]
                    K_big = np.kron(K_Dd_Xx_j, self.K_y)
                    theta_ate = K_big @ self.alpha
                    predictions_ate[j, :] = theta_ate
                return predictions_ate
            else:
                # OVK with V => K_XX_DD_VV
                # ATE
                for j in range(n_new_X):
                    K_Dd_Xx_j = K_Dd_Xx_new[j, :]
                    K_big = np.kron(K_Dd_Xx_j, self.K_y)
                    theta_ate = K_big @ self.alpha
                    predictions_ate[j, :] = theta_ate

                # CATE only if V_new is provided
                if do_cate:
                    if V_new.ndim == 1:
                        V_new = V_new.reshape(-1, 1)
                    K_VV_new = self._rbf_kernel(V_new, self.V_train)
                    n_new_V = V_new.shape[0]
                    predictions_cate = np.zeros((n_new_X, n_new_V, self.n_timepoints))

                    for j in range(n_new_X):
                        K_Dd_Xx_j = K_Dd_Xx_new[j, :]
                        for i in range(n_new_V):
                            K_Vv_i = K_VV_new[i, :]
                            K_combined = np.multiply(K_Dd_Xx_j, K_Vv_i)
                            K_big = np.kron(K_combined, self.K_y)
                            theta_cate = K_big @ self.alpha
                            predictions_cate[j, i, :] = theta_cate
                    return predictions_ate, predictions_cate
                else:
                    return predictions_ate
