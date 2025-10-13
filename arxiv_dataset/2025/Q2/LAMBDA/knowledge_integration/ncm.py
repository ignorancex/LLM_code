from .knw import knw


class Nearest_Correlation_Matrix(knw):

    def __init__(self):
        super().__init__()
        self.name = 'nearest_correlation_matrix'
        self.description = 'The function calculates the nearest correlation matrix using the quadratically convergent newton method. Acceptable parameters: Sigma, b>0, tau>=0, and tol (tolerance error) For the correlation matrix problem, set b = np.ones((n,1)).'
        self.core_function = 'core'
        self.runnable_function = 'runnable'
        self.mode = 'core'

    def core(self):
        return """
        n = 3000
        data_g_test = scipy.randn(n, n)
        data_g_test = (data_g_test + data_g_test.transpose()) / 2.0
        data_g_test = data_g_test - np.diag(np.diag(data_g_test)) + np.eye(n)
        b = np.ones((n, 1))
        tau = 0
        tol = 1.0e-6
        [x_test_result, y_test_result] = NearestCorrelationMatrix(data_g_test, b, tau, tol)
        print("The x_test_result: \n", x_test_result)
        print()
        print("The y_test_result: \n", y_test_result)
        """

    def runnable(self):
        code = r"""
        import numpy as np
        import scipy.io as sio
        import scipy
        import sys
        import time

        def NearestCorrelationMatrix(g_input, b_input=None, tau=None, tol=None):
            print('-- Semismooth Newton-CG method starts --\n')
            [n, m] = g_input.shape
            g_input = g_input.copy()
            t0 = time.time()  # time start
            g_input = (g_input + g_input.transpose()) / 2.0
            b_g = np.ones((n, 1))
            error_tol = 1.0e-6
            if b_input is None:
                tau = 0
            elif tau is None:
                b_g = b_input.copy()
                tau = 0
            elif tol is None:
                b_g = b_input.copy() - tau * np.ones((n, 1))
                g_input = g_input - tau * np.eye(n, n)
            else:
                b_g = b_input.copy() - tau * np.ones((n, 1))
                g_input = g_input - tau * np.eye(n, n)
                error_tol = np.maximum(1.0e-12, tol)

            res_b = np.zeros((300, 1))
            norm_b0 = np.linalg.norm(b_g)
            y = np.zeros((n, 1))
            f_y = np.zeros((n, 1))
            k = 0
            f_eval = 0
            iter_whole = 200
            iter_inner = 20  # maximum number of line search in Newton method
            maxit = 200  # maximum number of iterations in PCG
            iterk = 0
            inner = 0
            tol_cg = 1.0e-2  # relative accuracy for CGs
            sigma_1 = 1.0e-4
            x0 = y.copy()
            prec_time = 0
            pcg_time = 0
            eig_time = 0
            c = np.ones((n, 1))
            d = np.zeros((n, 1))
            val_g = np.sum((g_input.astype(float)) * (g_input.astype(float)))
            val_g = val_g * 0.5
            x_result = g_input + np.diagflat(y)
            x_result = (x_result + x_result.transpose()) / 2.0
            eig_time0 = time.time()
            [p_x, lamb] = my_mexeig(x_result)
            eig_time = eig_time + (time.time() - eig_time0)
            [f_0, f_y] = my_gradient(y, lamb, p_x, b_g, n)
            initial_f = val_g - f_0
            x_result = my_pca(x_result, lamb, p_x, b_g, n)
            val_obj = np.sum(((x_result - g_input) * (x_result - g_input))) / 2.0
            gap = (val_obj - initial_f) / (1.0 + np.abs(initial_f) + np.abs(val_obj))
            f = f_0.copy()
            f_eval = f_eval + 1
            b_input = b_g - f_y
            norm_b = np.linalg.norm(b_input)
            time_used = time.time() - t0

            print('Newton-CG: Initial Dual objective function value: %s \n' % initial_f)
            print('Newton-CG: Initial Primal objective function value: %s \n' % val_obj)
            print('Newton-CG: Norm of Gradient: %s \n' % norm_b)
            print('Newton-CG: computing time used so far: %s \n' % time_used)

            omega_12 = my_omega_mat(p_x, lamb, n)
            x0 = y.copy()

            while np.abs(gap) > error_tol and norm_b / (1 + norm_b0) > error_tol and k < iter_whole:
                prec_time0 = time.time()
                c = my_precond_matrix(omega_12, p_x, n)
                prec_time = prec_time + (time.time() - prec_time0)

                pcg_time0 = time.time()
                [d, flag, relres, iterk] = my_pre_cg(b_input, tol_cg, maxit, c, omega_12, p_x, n)
                pcg_time = pcg_time + (time.time() - pcg_time0)
                print('Newton-CG: Number of CG Iterations=== %s \n' % iterk)
                if flag == 1:
                    print('=== Not a completed Newton-CG step === \n')

                slope = np.dot((f_y - b_g).transpose(), d)

                y = (x0 + d).copy()
                x_result = g_input + np.diagflat(y)
                x_result = (x_result + x_result.transpose()) / 2.0
                eig_time0 = time.time()
                [p_x, lamb] = my_mexeig(x_result)
                eig_time = eig_time + (time.time() - eig_time0)
                [f, f_y] = my_gradient(y, lamb, p_x, b_g, n)

                k_inner = 0
                while k_inner <= iter_inner and f > f_0 + sigma_1 * (np.power(0.5, k_inner)) * slope + 1.0e-6:
                    k_inner = k_inner + 1
                    y = x0 + (np.power(0.5, k_inner)) * d
                    x_result = g_input + np.diagflat(y)
                    x_result = (x_result + x_result.transpose()) / 2.0
                    eig_time0 = time.time()
                    [p_x, lamb] = my_mexeig(x_result)
                    eig_time = eig_time + (time.time() - eig_time0)
                    [f, f_y] = my_gradient(y, lamb, p_x, b_g, n)

                f_eval = f_eval + k_inner + 1
                x0 = y.copy()
                f_0 = f.copy()
                val_dual = val_g - f_0
                x_result = my_pca(x_result, lamb, p_x, b_g, n)
                val_obj = np.sum((x_result - g_input) * (x_result - g_input)) / 2.0
                gap = (val_obj - val_dual) / (1 + np.abs(val_dual) + np.abs(val_obj))
                print('Newton-CG: The relative duality gap: %s \n' % gap)
                print('Newton-CG: The Dual objective function value: %s \n' % val_dual)
                print('Newton-CG: The Primal objective function value: %s \n' % val_obj)

                b_input = b_g - f_y
                norm_b = np.linalg.norm(b_input)
                time_used = time.time() - t0
                rel_norm_b = norm_b / (1 + norm_b0)
                print('Newton-CG: Norm of Gradient: %s \n' % norm_b)
                print('Newton-CG: Norm of Relative Gradient: %s \n' % rel_norm_b)
                print('Newton-CG: Computing time used so for %s \n' % time_used)
                res_b[k] = norm_b
                k = k + 1
                omega_12 = my_omega_mat(p_x, lamb, n)

            position_rank = np.maximum(lamb, 0) > 0
            rank_x = (np.maximum(lamb, 0)[position_rank]).size
            final_f = val_g - f
            x_result = x_result + tau * (np.eye(n))
            time_used = time.time() - t0
            print('\n')

            print('Newton-CG: Number of iterations: %s \n' % k)
            print('Newton-CG: Number of Funtion Evaluation:  =========== %s\n' % f_eval)
            print('Newton-CG: Final Dual Objective Function value: ========= %s\n' % final_f)
            print('Newton-CG: Final Primal Objective Function value: ======= %s \n' % val_obj)
            print('Newton-CG: The final relative duality gap: %s \n' % gap)
            print('Newton-CG: The rank of the Optimal Solution - tau*I: %s \n' % rank_x)
            print('Newton-CG: computing time for computing preconditioners: %s \n' % prec_time)
            print('Newton-CG: computing time for linear system solving (cgs time): %s \n' % pcg_time)
            print('Newton-CG: computing time for eigenvalue decompositions: =============== %s \n' % eig_time)
            print('Newton-CG: computing time used for equal weight calibration ============ %s \n' % time_used)

            return x_result, y

        def my_gradient(y_input, lamb, p_input, b_0, n):
            f = 0.0
            Fy = np.zeros((n, 1))
            p_input_copy = (p_input.copy()).transpose()
            for i in range(0, n):
                p_input_copy[i, :] = ((np.maximum(lamb[i], 0).astype(float)) ** 0.5) * p_input_copy[i, :]

            for i in range(0, n):
                Fy[i] = np.sum(p_input_copy[:, i] * p_input_copy[:, i])

            for i in range(0, n):
                f = f + np.square((np.maximum(lamb[i], 0)))

            f = 0.5 * f - np.dot(b_0.transpose(), y_input)

            return f, Fy


        # use PCA to generate a primal feasible solution checked

        def my_pca(x_input, lamb, p_input, b_0, n):
            x_pca = x_input
            lamb = np.asarray(lamb)
            lp = lamb > 0
            r = lamb[lp].size
            if r == 0:
                x_pca = np.zeros((n, n))
            elif r == n:
                x_pca = x_input
            elif r < (n / 2.0):
                lamb1 = lamb[lp].copy()
                lamb1 = lamb1.transpose()
                lamb1 = np.sqrt(lamb1.astype(float))
                P1 = p_input[:, 0:r].copy()
                if r > 1:
                    P1 = np.dot(P1, np.diagflat(lamb1))
                    x_pca = np.dot(P1, P1.transpose())
                else:
                    x_pca = np.dot(np.dot(np.square(lamb1), P1), P1.transpose())

            else:
                lamb2 = -lamb[r:n].copy()
                lamb2 = np.sqrt(lamb2.astype(float))
                p_2 = p_input[:, r:n]
                p_2 = np.dot(p_2, np.diagflat(lamb2))
                x_pca = x_pca + np.dot(p_2, p_2.transpose())

            # To make x_pca positive semidefinite with diagonal elements exactly b0
            d = x_pca.diagonal()
            d = d.reshape((d.size, 1))
            d = np.maximum(d, b_0.reshape(d.shape))
            x_pca = x_pca - np.diagflat(x_pca.diagonal()) + np.diagflat(d)
            d = d.astype(float) ** (-0.5)
            d = d * ((np.sqrt(b_0.astype(float))).reshape(d.shape))
            x_pca = x_pca * (np.dot(d, d.reshape(1, d.size)))

            return x_pca


        # end of PCA

        # To generate the first order difference of lambda
        # To generate the first order essential part of d


        def my_omega_mat(p_input, lamb, n):
            idx_idp = np.where(lamb > 0)
            idx_idp = idx_idp[0]
            idx_idm = np.setdiff1d(range(0, n), idx_idp)
            n = lamb.size
            r = idx_idp.size
            if r > 0:
                if r == n:
                    omega_12 = np.ones((n, n))
                else:
                    s = n - r
                    dp = lamb[0:r].copy()
                    dp = dp.reshape(dp.size, 1)
                    dn = lamb[r:n].copy()
                    dn = dn.reshape((dn.size, 1))
                    omega_12 = np.dot(dp, np.ones((1, s)))
                    omega_12 = omega_12 / (np.dot(np.abs(dp), np.ones((1, s))) + np.dot(np.ones((r, 1)), abs(dn.transpose())))
                    omega_12 = omega_12.reshape((r, s))

            else:
                omega_12 = np.array([])

            return omega_12


        # End of my_omega_mat


        # To generate Jacobian


        def my_jacobian_matrix(x, omega_12, p_input, n):
            x_result = np.zeros((n, 1))
            [r, s] = omega_12.shape
            if r > 0:
                hmat_1 = p_input[:, 0:r].copy()
                if r < n / 2.0:
                    i = 0
                    while i < n:
                        hmat_1[i, :] = x[i] * hmat_1[i, :]
                        i = i + 1

                    omega_12 = omega_12 * (np.dot(hmat_1.transpose(), p_input[:, r:n]))
                    hmat = np.dot(hmat_1.transpose(), np.dot(p_input[:, 0:r], p_input[:, 0:r].transpose()))
                    hmat = hmat + np.dot(omega_12, p_input[:, r:n].transpose())
                    hmat = np.vstack((hmat, np.dot(omega_12.transpose(), p_input[:, 0:r].transpose())))
                    i = 0
                    while i < n:
                        x_result[i] = np.dot(p_input[i, :], hmat[:, i])
                        x_result[i] = x_result[i] + 1.0e-10 * x[i]
                        i = i + 1

                else:
                    if r == n:
                        x_result = 1.0e-10 * x
                    else:
                        hmat_2 = p_input[:, r:n].copy()
                        i = 0
                        while i < n:
                            hmat_2[i, :] = x[i] * hmat_2[i, :]
                            i = i + 1

                        omega_12 = np.ones((r, s)) - omega_12
                        omega_12 = omega_12 * (np.dot(p_input[:, 0:r].transpose(), hmat_2))
                        hmat = np.dot(p_input[:, r:n].transpose(), hmat_2)
                        hmat = np.dot(hmat, p_input[:, r:n].transpose())
                        hmat = hmat + np.dot(omega_12.transpose(), p_input[:, 0:r].transpose())
                        hmat = np.vstack((np.dot(omega_12, p_input[:, r:n].transpose()), hmat))
                        i = 0
                        while i < n:
                            x_result[i] = np.dot(-p_input[i, :], hmat[:, i])
                            x_result[i] = x[i] + x_result[i] + 1.0e-10 * x[i]
                            i = i + 1

            return x_result


        # end of Jacobian
        # PCG Method


        def my_pre_cg(b, tol, maxit, c, omega_12, p_input, n):
            # Initializations
            r = b.copy()
            r = r.reshape(r.size, 1)
            c = c.reshape(c.size, 1)
            n2b = np.linalg.norm(b)
            tolb = tol * n2b
            p = np.zeros((n, 1))
            flag = 1
            iterk = 0
            relres = 1000
            # Precondition
            z = r / c
            rz_1 = np.dot(r.transpose(), z)
            rz_2 = 1
            d = z.copy()
            # d = d.reshape(z.shape)
            # CG Iteration
            for k in range(0, maxit):
                if k > 0:
                    beta = rz_1 / rz_2
                    d = z + beta * d

                w = my_jacobian_matrix(d, omega_12, p_input, n)
                denom = np.dot(d.transpose(), w)
                iterk = k + 1
                relres = np.linalg.norm(r) / n2b
                if denom <= 0:
                    ss = 0  # don't know the usage, check the paper
                    p = d / np.linalg.norm(d)
                    break
                else:
                    alpha = rz_1 / denom
                    p = p + alpha * d
                    r = r - alpha * w

                z = r / c
                if np.linalg.norm(r) <= tolb:  # exit if hmat p = b solved in relative error tolerance
                    iterk = k + 1
                    relres = np.linalg.norm(r) / n2b
                    flag = 0
                    break

                rz_2 = rz_1
                rz_1 = np.dot(r.transpose(), z)

            return p, flag, relres, iterk


        # end of pre_cg

        # to generate the diagonal preconditioner


        def my_precond_matrix(omega_12, p_input, n):
            [r, s] = omega_12.shape
            c = np.ones((n, 1))
            if r > 0:
                if r < n / 2.0:
                    hmat = (p_input.copy()).transpose()
                    hmat = hmat * hmat
                    hmat_12 = np.dot(hmat[0:r, :].transpose(), omega_12)
                    d = np.ones((r, 1))
                    for i in range(0, n):
                        c_temp = np.dot(d.transpose(), hmat[0:r, i])
                        c_temp = c_temp * hmat[0:r, i]
                        c[i] = np.sum(c_temp)
                        c[i] = c[i] + 2.0 * np.dot(hmat_12[i, :], hmat[r:n, i])
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8

                else:
                    if r < n:
                        hmat = (p_input.copy()).transpose()
                        hmat = hmat * hmat
                        omega_12 = np.ones((r, s)) - omega_12
                        hmat_12 = np.dot(omega_12, hmat[r:n, :])
                        d = np.ones((s, 1))
                        dd = np.ones((n, 1))

                        for i in range(0, n):
                            c_temp = np.dot(d.transpose(), hmat[r:n, i])
                            c[i] = np.sum(c_temp * hmat[r:n, i])
                            c[i] = c[i] + 2.0 * np.dot(hmat[0:r, i].transpose(), hmat_12[:, i])
                            alpha = np.sum(hmat[:, i])
                            c[i] = alpha * np.dot(hmat[:, i].transpose(), dd) - c[i]
                            if c[i] < 1.0e-8:
                                c[i] = 1.0e-8

            return c


        # end of precond_matrix


        # my_issorted()

        def my_issorted(x_input, flag):
            n = x_input.size
            tf_value = False
            if n < 2:
                tf_value = True
            else:
                if flag == 1:
                    i = 0
                    while i < n - 1:
                        if x_input[i] <= x_input[i + 1]:
                            i = i + 1
                        else:
                            break

                    if i == n - 1:
                        tf_value = True
                    elif i < n - 1:
                        tf_value = False

                elif flag == -1:
                    i = n - 1
                    while i > 0:
                        if x_input[i] <= x_input[i - 1]:
                            i = i - 1
                        else:
                            break

                    if i == 0:
                        tf_value = True
                    elif i > 0:
                        tf_value = False

            return tf_value


        # end of my_issorted()


        def my_mexeig(x_input):
            [n, m] = x_input.shape
            [lamb, p_x] = np.linalg.eigh(x_input)
            # lamb = lamb.reshape((lamb.size, 1))
            p_x = p_x.real
            lamb = lamb.real
            if my_issorted(lamb, 1):
                lamb = lamb[::-1]
                p_x = np.fliplr(p_x)
            elif my_issorted(lamb, -1):
                return p_x, lamb
            else:
                idx = np.argsort(-lamb)
                # lamb_old = lamb   # add for debug
                lamb = lamb[idx]
                # p_x_old = p_x   add for debug
                p_x = p_x[:, idx]

            lamb = lamb.reshape((n, 1))
            p_x = p_x.reshape((n, n))

            return p_x, lamb

        """
        return code


if __name__ == '__main__':
    ncm = Nearest_Correlation_Matrix()
    print(ncm.get_core_function())
    print(ncm.runnable())
