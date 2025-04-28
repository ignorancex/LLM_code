from scipy.special import eval_hermite
from Helper import *

impath = "../plots/images_synthetic/"
os.makedirs(impath, exist_ok=True)


class synthetic_sup:
    def __init__(self, training_samples=[], testing_sample=[], nmodes=8, spod_iter=300, plot_offline_data=False):
        self.Nx = 500  # number of grid points in x
        self.Ny = 1  # number of grid points in y
        self.Nt = 500  # numer of time intervals

        self.NumFrames = 2

        self.T = 1  # total time
        self.L = 1  # total domain size
        self.nmodes = nmodes  # reduction of singular values
        self.D = self.nmodes

        self.x = np.arange(-self.Nx // 2, self.Nx // 2) / self.Nx * self.L
        self.t = np.arange(0, self.Nt) / self.Nt * self.T

        [self.Xgrid, self.Tgrid] = np.meshgrid(self.x, self.t)
        self.Xgrid = self.Xgrid.T
        self.Tgrid = self.Tgrid.T

        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        c = 1

        print("#############################################")
        print("Synthetic data checks....")
        ##########################################
        # Create training data
        self.mu_vecs_train = np.asarray(training_samples)
        self.Nsamples_train = np.size(self.mu_vecs_train)
        self.q_train, q1_train, q2_train, self.shifts_train, self.params_train, self.trafos_train = \
            self.create_data(self.mu_vecs_train)
        ##########################################
        # Create testing data
        self.mu_vecs_test = np.asarray(testing_sample)
        self.Nsamples_test = np.size(self.mu_vecs_test)
        self.q_test, self.q1_test, self.q2_test, self.shifts_test, self.params_test, self.trafos_test = \
            self.create_data(self.mu_vecs_test)

        ##########################################
        # Calculate the transformation interpolation error
        interp_err = give_interpolation_error(self.q_train, self.trafos_train[0])
        print("Check 1...")
        print("Transformation interpolation error =  %4.4e " % interp_err)

        # Calculate the time amplitudes
        qmat = np.reshape(self.q_train, [-1, self.Nsamples_train * self.Nt])
        [N, M] = np.shape(qmat)
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.005
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 10
        ret = shifted_rPCA(self.q_train, self.trafos_train, nmodes_max=np.max(self.D) + 20, eps=1e-16, Niter=spod_iter,
                           use_rSVD=True, mu=mu0, lambd=lambd0)
        sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist

        ###########################################
        # relative offline error for training data (srPCA error)
        err_full = np.linalg.norm(self.q_train - qtilde) / np.linalg.norm(self.q_train)
        print("Check 2...")
        print("Error for full sPOD recons. is {}".format(err_full))

        ###########################################
        # Calculate the time amplitudes for training data
        self.U_list = []
        self.TA_interp_list = []
        TA_training_list = []
        qtrunc = 0
        cnt = 0
        for frame in sPOD_frames:
            VT = frame.modal_system["VT"][:self.D, :]
            S = frame.modal_system["sigma"][:self.D]
            VT = np.diag(S) @ VT
            Nmodes = self.D
            amplitudes = [np.reshape(VT[n, :], [self.Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            self.TA_interp_list.append(amplitudes)
            TA_training_list.append(VT)
            self.U_list.append(frame.modal_system["U"][:, :self.D])

            qtrunc += self.trafos_train[cnt].apply(self.U_list[cnt] @ TA_training_list[cnt])
            cnt = cnt + 1

        err_trunc = np.linalg.norm(self.q_train - qtrunc) / np.linalg.norm(self.q_train)
        print("Error for truncated sPOD recons. is {}".format(err_trunc))

        ###########################################
        # Generate data for the POD-NN for comparison
        U, S, VT = np.linalg.svd(np.squeeze(self.q_train), full_matrices=False)
        self.U_POD_TRAIN = U[:, :self.NumFrames * self.D + self.NumFrames]
        self.TA_POD_TRAIN = np.diag(S[:self.NumFrames * self.D + self.NumFrames]) @ \
                            VT[:self.NumFrames * self.D + self.NumFrames, :]
        self.TA_POD_TEST = self.U_POD_TRAIN.transpose() @ self.q_test

        ###########################################
        # data for the NN
        amplitudes_train = np.concatenate(TA_training_list, axis=0)
        time_amplitudes_1_test = self.U_list[0][:, :self.D].transpose() @ self.q1_test
        time_amplitudes_2_test = self.U_list[1][:, :self.D].transpose() @ self.q2_test
        amplitudes_test = np.concatenate((time_amplitudes_1_test, time_amplitudes_2_test), axis=0)

        self.TA_TRAIN = amplitudes_train
        self.SHIFTS_TRAIN = [self.shifts_train[0], self.shifts_train[1]]
        self.PARAMS_TRAIN = self.params_train
        self.TA_TEST = amplitudes_test
        self.SHIFTS_TEST = [self.shifts_test[0], self.shifts_test[1]]
        self.PARAMS_TEST = self.params_test

        if plot_offline_data:
            # Plot all the variables required
            q1_spod_frame = sPOD_frames[0].build_field()
            q2_spod_frame = sPOD_frames[1].build_field()
            self.plot_FOM_data(self.q_train, q1_train, q2_train, self.Nsamples_train)
            self.plot_sPODframes(self.q_train, qtilde, q1_spod_frame, q2_spod_frame)

    def create_data(self, mu_vecs):

        # gauss hermite polynomials of order n
        psi = lambda n, x: np.exp(-x ** 2 / 2) * eval_hermite(n, x)
        Nsamples = np.size(mu_vecs)
        w = 0.015 * self.L
        Nt = np.size(self.Tgrid, 1)

        qs1 = []
        qs2 = []
        for k in range(Nsamples):  # loop over all possible mu vectors
            q1 = np.zeros_like(self.Xgrid)
            q2 = np.zeros_like(self.Xgrid)
            for n in range(self.D):  # loop over all components of the vector
                q1 += (1 + np.exp(-2 * n * self.Tgrid)) * mu_vecs[k] * np.cos(
                    - 2 * np.pi * self.Tgrid / self.T * (n + 1)) * psi(n, (
                        self.Xgrid + 0.1 * self.L) / w)
                q2 += (1 + np.exp(-2 * n * self.Tgrid)) * mu_vecs[k] * np.cos(
                    - 2 * np.pi * self.Tgrid / self.T * (n + 1)) * psi(n, (
                        self.Xgrid - 0.1 * self.L) / w)

            qs1.append(q1)
            qs2.append(-q2)

        q1 = np.concatenate(qs1, axis=1)
        q2 = np.concatenate(qs2, axis=1)
        q_frames = [q1, q2]

        shifts = [np.asarray([mu_vecs[n] * self.t]) for n in range(Nsamples)]

        shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
        data_shape = [self.Nx, 1, 1, Nt * Nsamples]
        trafos = [
            transforms(data_shape, [self.L], shifts=np.squeeze(shifts[0]), dx=[self.dx], use_scipy_transform=False,
                       interp_order=5),
            transforms(data_shape, [self.L], shifts=np.squeeze(shifts[1]), dx=[self.dx], use_scipy_transform=False,
                       interp_order=5)]

        q = 0
        for trafo, qf in zip(trafos, q_frames):
            q += trafo.apply(qf)

        # Parameter matrix
        p = [np.squeeze(np.asarray([[self.t], [np.ones_like(self.t) * mu]])) for mu in mu_vecs]
        p = np.concatenate(p, axis=1)

        return q, q1, q2, shifts, p, trafos

    def OnlinePredictionAnalysis(self, TA_sPOD_pred, shifts_sPOD_pred, TA_POD_pred,
                                 plot_online=False, test_type=None):

        if test_type['typeOfTest'] == "query":
            plot_online = False
            test_sample = test_type['test_sample']
            self.q_test = self.q_test[:, test_sample][..., np.newaxis]
            for frame in range(self.NumFrames):
                self.shifts_train[frame] = np.asarray([self.shifts_train[frame][:, i * self.Nt + test_sample]
                                                      for i in range(self.Nsamples_train)]).transpose()


        print("#############################################")
        print('Online Error checks')
        Nx = len(self.x)
        Nt = TA_sPOD_pred.shape[1]  # len(self.t)
        data_shape = [Nx, 1, 1, Nt]
        dx = self.x[1] - self.x[0]
        TA_sPOD_pred_1 = TA_sPOD_pred[:self.D, :]
        TA_sPOD_pred_2 = TA_sPOD_pred[self.D:2 * self.D, :]
        shifts_sPOD_pred_1 = shifts_sPOD_pred[0, :]
        shifts_sPOD_pred_2 = shifts_sPOD_pred[1, :]

        ###########################################
        # Implement the interpolation to find the online prediction
        shifts_list_interpolated = []
        cnt = 0
        for frame in range(self.NumFrames):
            shifts = np.reshape(self.shifts_train[cnt], [self.Nsamples_train, Nt]).T
            shifts_list_interpolated.append(shifts)
            cnt = cnt + 1

        DELTA_PRED_FRAME_WISE = my_delta_interpolate(shifts_list_interpolated, self.mu_vecs_train, self.mu_vecs_test)
        Nmodes = [self.D, self.D]
        trafos_interp = [
            transforms(data_shape, [self.L], shifts=DELTA_PRED_FRAME_WISE[0], dx=[self.dx], use_scipy_transform=False,
                       interp_order=5),
            transforms(data_shape, [self.L], shifts=DELTA_PRED_FRAME_WISE[1], dx=[self.dx], use_scipy_transform=False,
                       interp_order=5)
        ]
        q_interp, TA_interp = my_interpolated_state(Nmodes, self.U_list, self.TA_interp_list,
                                                    self.mu_vecs_train,
                                                    self.Nx, self.Ny, Nt,
                                                    self.mu_vecs_test, trafos_interp)
        ###########################################

        # Shifts error
        num1 = np.linalg.norm(self.SHIFTS_TEST[0] - shifts_sPOD_pred_1.flatten())
        den1 = np.linalg.norm(self.SHIFTS_TEST[0])
        num2 = np.linalg.norm(self.SHIFTS_TEST[1] - shifts_sPOD_pred_2.flatten())
        den2 = np.linalg.norm(self.SHIFTS_TEST[1])
        num3 = np.linalg.norm(self.SHIFTS_TEST[0] - DELTA_PRED_FRAME_WISE[0])
        den3 = np.linalg.norm(self.SHIFTS_TEST[0])
        num4 = np.linalg.norm(self.SHIFTS_TEST[1] - DELTA_PRED_FRAME_WISE[1])
        den4 = np.linalg.norm(self.SHIFTS_TEST[1])
        print('Check 1...')
        print("Relative error indicator (sPOD-NN) for shift: 1 is {}".format(num1 / den1))
        print("Relative error indicator (sPOD-NN) for shift: 2 is {}".format(num2 / den2))
        print("Relative error indicator (sPOD-I) for shift: 1 is {}".format(num3 / den3))
        print("Relative error indicator (sPOD-I) for shift: 2 is {}".format(num4 / den4))

        # Time amplitudes error
        TA_test_1 = self.TA_TEST[:self.D, :]
        TA_test_2 = self.TA_TEST[self.D:2 * self.D, :]
        num1 = np.linalg.norm(TA_test_1 - TA_sPOD_pred_1)
        den1 = np.linalg.norm(TA_test_1)
        num2 = np.linalg.norm(TA_test_2 - TA_sPOD_pred_2)
        den2 = np.linalg.norm(TA_test_2)
        num3 = np.linalg.norm(np.squeeze(TA_test_1) - TA_interp[0])
        den3 = np.linalg.norm(np.squeeze(TA_test_1))
        num4 = np.linalg.norm(np.squeeze(TA_test_2) - TA_interp[1])
        den4 = np.linalg.norm(np.squeeze(TA_test_2))
        num5 = np.linalg.norm(self.TA_POD_TEST - TA_POD_pred)
        den5 = np.linalg.norm(self.TA_POD_TEST)
        print('Check 2...')
        print("Relative time amplitude error indicator (sPOD-NN) for frame: 1 is {}".format(num1 / den1))
        print("Relative time amplitude error indicator (sPOD-NN) for frame: 2 is {}".format(num2 / den2))
        print("Relative time amplitude error indicator (sPOD-I) for frame: 1 is {}".format(num3 / den3))
        print("Relative time amplitude error indicator (sPOD-I) for frame: 2 is {}".format(num4 / den4))
        print("Relative time amplitude error indicator (POD-NN) is {}".format(num5 / den5))

        q_sPOD_pred_1 = self.U_list[0][:, :self.D] @ TA_sPOD_pred_1
        q_sPOD_pred_2 = self.U_list[1][:, :self.D] @ TA_sPOD_pred_2
        # Total reconstructed error
        q_sPOD_recon = 0
        NumFrames = 2
        q_pred = [np.reshape(q_sPOD_pred_1, newshape=data_shape), np.reshape(q_sPOD_pred_2, newshape=data_shape)]
        trafos = [
            transforms(data_shape, [self.L], shifts=shifts_sPOD_pred_1, dx=[self.dx],
                       use_scipy_transform=False,
                       interp_order=5),
            transforms(data_shape, [self.L], shifts=shifts_sPOD_pred_2, dx=[self.dx],
                       use_scipy_transform=False,
                       interp_order=5)]
        for frame in range(NumFrames):
            q_sPOD_recon += trafos[frame].apply(q_pred[frame])
        q_POD_recon = self.U_POD_TRAIN @ TA_POD_pred

        self.q_test = np.squeeze(self.q_test)
        q_sPOD_recon = np.squeeze(q_sPOD_recon)
        q_POD_recon = np.squeeze(q_POD_recon)
        q_interp = np.squeeze(q_interp)

        num1 = np.linalg.norm(self.q_test - q_sPOD_recon)
        den1 = np.linalg.norm(self.q_test)

        num2 = np.linalg.norm(self.q_test - q_POD_recon)
        den2 = np.linalg.norm(self.q_test)

        num3 = np.linalg.norm(self.q_test - q_interp)
        den3 = np.linalg.norm(self.q_test)

        print('Check 3...')
        print("Relative reconstruction error indicator for full snapshot (sPOD-NN) is {}".format(num1 / den1))
        print("Relative reconstruction error indicator for full snapshot (sPOD-I) is {}".format(num3 / den3))
        print("Relative reconstruction error indicator for full snapshot (POD-NN) is {}".format(num2 / den2))

        if test_type['typeOfTest'] != "query":
            one = self.q_test - q_sPOD_recon
            num1 = np.sqrt(np.einsum('ij,ij->j', one, one))
            den1 = np.sqrt(np.sum(np.einsum('ij,ij->j', self.q_test, self.q_test)) / self.Nt)

            two = self.q_test - q_POD_recon
            num2 = np.sqrt(np.einsum('ij,ij->j', two, two))

            three = self.q_test - q_interp
            num3 = np.sqrt(np.einsum('ij,ij->j', three, three))

            rel_err_sPOD = num1 / den1
            rel_err_POD = num2 / den1
            rel_err_interp = num3 / den1

            errors = [rel_err_sPOD, rel_err_POD, rel_err_interp]
        else:
            errors = [np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt)]

        if plot_online:
            # Plot the online prediction data
            self.plot_timeamplitudes_shifts_Pred(TA_sPOD_pred_1, TA_test_1, TA_sPOD_pred_2,
                                                 TA_test_2, TA_POD_pred, TA_interp, shifts_sPOD_pred_1,
                                                 shifts_sPOD_pred_2, DELTA_PRED_FRAME_WISE)
            self.plot_recons_snapshot(q_sPOD_recon, q_POD_recon, q_interp)

        return errors

    def plot_FOM_data(self, q_train, q1_train, q2_train, Nsamples_train):
        Nx = self.Nx
        Nt = self.Nt
        fig, axs = plt.subplots(3, Nsamples_train, sharey=True, sharex=True, figsize=(10, 6), num=1)
        plt.subplots_adjust(wspace=0)
        qmin = np.min(q_train)
        qmax = np.max(q_train)
        for k in range(0, Nsamples_train):
            kw = k
            axs[0, kw].pcolormesh(q_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
            axs[0, kw].set_title(r'${\mu}_{' + str(k) + '}$')
            axs[0, kw].set_yticks([0, Nx // 2, Nx])
            axs[0, kw].set_xticks([0, Nt // 2, Nt])
            axs[0, kw].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
            axs[1, kw].pcolormesh(q1_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
            im = axs[2, kw].pcolormesh(q2_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)

        # axs[0, 0].set_ylabel(r"$q$")
        # axs[1, 0].set_ylabel(r"$q^1$")
        # axs[2, 0].set_ylabel(r"$q^2$")
        axs[0, 0].set_xticklabels(["", r"$T/2$", r"$T$"])
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.25, 0.01, 0.5])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(impath + "synthetic_" + "training" + '.png', dpi=300, transparent=True)

    def plot_sPODframes(self, q_train, qtilde, q1_spod_frame, q2_spod_frame):
        qmin = np.min(q_train)
        qmax = np.max(q_train)
        Nx = self.Nx
        Nt = self.Nt
        fig, axs = plt.subplots(1, 3, num=3, sharey=True, sharex=True, figsize=(10, 3))
        bottom, top = 0.3, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0)
        # 1. frame
        k_frame = 0
        im = axs[0].pcolormesh(q1_spod_frame[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[0].set_title(r"$Q^" + str(k_frame + 1) + "$")
        axs[0].set_yticks([0, Nx // 2, Nx])
        axs[0].set_xticks([0, Nt // 2, Nt])
        axs[0].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
        axs[0].set_xticklabels(["", r"$T/2$", r"$T$"])
        # 2. frame
        k_frame = 1
        im = axs[1].pcolormesh(q2_spod_frame[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[1].set_title(r"$Q^" + str(k_frame + 1) + "$")
        axs[1].set_yticks([0, Nx // 2, Nx])
        axs[1].set_xticks([0, Nt // 2, Nt])
        axs[1].set_xticklabels(["", r"$T/2$", r"$T$"])
        # Reconstruction
        im = axs[2].pcolormesh(qtilde[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[2].set_title(r"$Q$")
        axs[2].set_yticks([0, Nx // 2, Nx])
        axs[2].set_xticks([0, Nt // 2, Nt])
        axs[2].set_xticklabels(["", r"$T/2$", r"$T$"])

        cbar_ax = fig.add_axes([0.85, bottom, 0.01, top - bottom])
        fig.colorbar(im, cax=cbar_ax)

        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")

        fig.savefig(impath + "synthetic_spod_frames" + ".png", dpi=300, transparent=True)

    def plot_timeamplitudes_shifts_Pred(self, TA_sPOD_pred_1, TA_test_1, TA_sPOD_pred_2,
                                        TA_test_2, TA_POD_pred, TA_interp, shifts_sPOD_pred_1,
                                        shifts_sPOD_pred_2, shifts_interp):

        fig = plt.figure(figsize=(12, 10), constrained_layout=True)

        # Create top/bottom subfigs
        (subfig_t, subfig_b) = fig.subfigures(2, 1, hspace=0.05, wspace=0.1)

        # put 3 axis in the top subfigure
        gs_t = subfig_t.add_gridspec(nrows=1, ncols=6)
        ax1 = subfig_t.add_subplot(gs_t[0, :2])
        ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharey=ax1)
        ax3 = subfig_t.add_subplot(gs_t[0, 4:], sharey=ax1)
        ax1.plot(self.t, TA_test_1[0, :], color="green", linestyle='-', label='actual')
        ax1.plot(self.t, TA_sPOD_pred_1[0, :], color="red", linestyle='--', label='sPOD-NN')
        ax1.plot(self.t, TA_interp[0][0, :], color="blue", linestyle='--', label='sPOD-I')
        ax1.set_xticks([0, self.t[-1] / 2, self.t[-1]])
        ax1.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax1.legend(loc='lower right')
        ax1.set_xlabel(r"(a)")
        ax1.grid()

        ax2.plot(self.t, TA_test_2[0, :], color="green", linestyle='-', label='actual')
        ax2.plot(self.t, TA_sPOD_pred_2[0, :], color="red", linestyle='--', label='sPOD-NN')
        ax2.plot(self.t, TA_interp[1][0, :], color="blue", linestyle='--', label='sPOD-I')
        ax2.set_xticks([0, self.t[-1] / 2, self.t[-1]])
        ax2.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax2.legend(loc='lower right')
        ax2.set_xlabel(r"(b)")
        ax2.grid()

        ax3.plot(self.t, self.TA_POD_TEST[0, :], color="green", linestyle='-', label='actual')
        ax3.plot(self.t, TA_POD_pred[0, :], color="black", linestyle='--', label='POD-NN')
        ax3.set_xticks([0, self.t[-1] / 2, self.t[-1]])
        ax3.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax3.legend(loc='upper right')
        ax3.set_xlabel(r"(c)")
        ax3.grid()

        subfig_t.supylabel(r"$a_i^{k}(t,\mu)$")
        subfig_t.supxlabel(r"time $t$")

        # put 2 axis in the bottom subfigure
        gs_b = subfig_b.add_gridspec(nrows=1, ncols=6)
        ax4 = subfig_b.add_subplot(gs_b[0, 1:3])
        ax5 = subfig_b.add_subplot(gs_b[0, 3:5])
        ax4.plot(self.t, self.SHIFTS_TEST[0].flatten(), color="green", linestyle='-', label='actual')
        ax4.plot(self.t, shifts_sPOD_pred_1.flatten(), color="red", linestyle='--', label='sPOD-NN')
        ax4.plot(self.t, shifts_interp[0], color="blue", linestyle='--', label='sPOD-I')
        ax4.set_xticks([0, self.t[-1] / 2, self.t[-1]])
        ax4.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax4.set_xlabel(r"(d)")
        ax4.grid()
        ax4.legend(loc='lower right')

        ax5.plot(self.t, self.SHIFTS_TEST[1].flatten(), color="green", linestyle='-', label='actual')
        ax5.plot(self.t, shifts_sPOD_pred_2.flatten(), color="red", linestyle='--', label='sPOD-NN')
        ax5.plot(self.t, shifts_interp[1], color="blue", linestyle='--', label='sPOD-I')
        ax5.set_xticks([0, self.t[-1] / 2, self.t[-1]])
        ax5.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax5.set_xlabel(r"(e)")
        ax5.grid()
        ax5.legend(loc='lower left')

        subfig_b.supxlabel(r"time $t$")
        subfig_b.supylabel(r"shifts $\underline{\Delta}^k$")

        save_fig(filepath=impath + "time_amplitudes_shifts_newplot_predicted", figure=fig)
        fig.savefig(impath + "time_amplitudes_shifts_newplot_predicted" + ".pdf", format='pdf',
                    dpi=200, transparent=True, bbox_inches="tight")

    def plot_recons_snapshot(self, q_sPOD_recon, q_POD_recon, q_interp):
        qmin = np.min(self.q_test)
        qmax = np.max(self.q_test)
        Nx = self.Nx
        Nt = self.Nt
        fig, axs = plt.subplots(1, 4, num=12, sharey=True, figsize=(12, 4))
        bottom, top = 0.3, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0)
        # Original
        im = axs[0].pcolormesh(self.q_test, cmap=cm, vmin=qmin, vmax=qmax)
        axs[0].set_title(r"$Q$")
        axs[0].set_yticks([0, Nx // 2, Nx])
        axs[0].set_xticks([0, Nt // 2, Nt])
        axs[0].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
        axs[0].set_xticklabels(["", r"$T/2$", r"$T$"])
        # Interpolated
        axs[1].pcolormesh(q_interp, cmap=cm, vmin=qmin, vmax=qmax)
        axs[1].set_title(r"$Q^" + "{\mathrm{sPOD-I}}$")
        axs[1].set_yticks([0, Nx // 2, Nx])
        axs[1].set_xticks([0, Nt // 2, Nt])
        axs[1].set_xticklabels(["", r"$T/2$", r"$T$"])
        # sPOD NN predicted
        axs[2].pcolormesh(q_sPOD_recon, cmap=cm, vmin=qmin, vmax=qmax)
        axs[2].set_yticks([0, Nx // 2, Nx])
        axs[2].set_xticks([0, Nt // 2, Nt])
        axs[2].set_title(r"$Q^" + "{\mathrm{sPOD-NN}}$")
        axs[2].set_xticklabels(["", r"$T/2$", r"$T$"])
        # POD NN predicted
        axs[3].pcolormesh(q_POD_recon, cmap=cm, vmin=qmin, vmax=qmax)
        axs[3].set_yticks([0, Nx // 2, Nx])
        axs[3].set_xticks([0, Nt // 2, Nt])
        axs[3].set_title(r"$Q^" + "{\mathrm{POD-NN}}$")
        axs[3].set_xticklabels(["", r"$T/2$", r"$T$"])

        cbar_ax = fig.add_axes([0.85, bottom, 0.01, top - bottom])
        fig.colorbar(im, cax=cbar_ax)

        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")

        fig.savefig(impath + "Snapshot_Comparison" + ".png", dpi=300, transparent=True)
