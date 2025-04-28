import time
from Helper import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

impath = "../plots/images_wildfire1D/"
os.makedirs(impath, exist_ok=True)

data_path = os.path.abspath(".") + '/wildfire_data/1D/'

cmap = 'YlOrRd'


# cmap = 'YlGn'


class wildfire1D_sup:
    def __init__(self, q_test, shifts_test, param_test_val, var):
        dat1_train = np.load(data_path + 'SnapShotMatrix540.npy')
        dat2_train = np.load(data_path + 'SnapShotMatrix550.npy')
        dat3_train = np.load(data_path + 'SnapShotMatrix560.npy')
        dat4_train = np.load(data_path + 'SnapShotMatrix570.npy')
        dat5_train = np.load(data_path + 'SnapShotMatrix580.npy')
        self.grid = np.load(data_path + '1D_Grid.npy', allow_pickle=True)
        self.x = self.grid[0]
        self.y = self.grid[1]
        self.t = np.load(data_path + 'Time.npy')
        delta1_train = np.load(data_path + 'Shifts540.npy')
        delta2_train = np.load(data_path + 'Shifts550.npy')
        delta3_train = np.load(data_path + 'Shifts560.npy')
        delta4_train = np.load(data_path + 'Shifts570.npy')
        delta5_train = np.load(data_path + 'Shifts580.npy')

        self.var = var
        self.Nx = np.size(self.x)
        self.Nt = np.size(self.t)

        self.param_test_val = param_test_val
        self.mu_vecs_test = np.asarray([self.param_test_val])

        self.q_test = q_test[self.var * self.Nx:(self.var + 1) * self.Nx, :]
        self.shifts_test = shifts_test

        q_train_total = np.concatenate((dat1_train, dat2_train, dat3_train, dat4_train, dat5_train), axis=1)
        self.q_train = q_train_total[var * self.Nx:(var + 1) * self.Nx, :]
        self.shifts_train = np.concatenate((delta1_train, delta2_train, delta3_train, delta4_train, delta5_train),
                                           axis=1)

        self.Nsamples_train = 5
        self.NumFrames = 3

        self.mu_vecs_train = np.asarray([540, 550, 560, 570, 580])
        self.params_train = [np.squeeze(np.asarray([[np.ones_like(self.t) * mu], [self.t]])) for mu in
                             self.mu_vecs_train]
        self.params_train = np.concatenate(self.params_train, axis=1)

    def run_sPOD(self, spod_iter):
        print("#############################################")
        print("sPOD run started....")
        ##########################################
        # Run srPCA
        dx = self.x[1] - self.x[0]
        L = [self.x[-1]]
        data_shape = [self.Nx, 1, 1, self.Nt * self.Nsamples_train]
        trafo_train_1 = transforms(data_shape, L, shifts=np.squeeze(self.shifts_train[0]).flatten(),
                                   dx=[dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafo_train_2 = transforms(data_shape, L, shifts=np.squeeze(self.shifts_train[1]).flatten(),
                                   trafo_type="identity", dx=[dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafo_train_3 = transforms(data_shape, L, shifts=np.squeeze(self.shifts_train[2]).flatten(),
                                   dx=[dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafos_train = [trafo_train_1, trafo_train_2, trafo_train_3]

        qmat = np.reshape(self.q_train, [-1, self.Nt * self.Nsamples_train])
        [N, M] = np.shape(qmat)
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.005
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 5

        ret_train = shifted_rPCA(self.q_train, trafos_train, nmodes_max=10, eps=1e-16, Niter=spod_iter, use_rSVD=True,
                                 mu=mu0, lambd=lambd0, dtol=1e-5)
        sPOD_frames_train, qtilde_train, rel_err_train = ret_train.frames, ret_train.data_approx, ret_train.rel_err_hist

        ###########################################
        # relative offline error for training wildfire_data (srPCA error)
        err_full = np.sqrt(np.mean(np.linalg.norm(self.q_train - qtilde_train, 2, axis=1) ** 2)) / \
                   np.sqrt(np.mean(np.linalg.norm(self.q_train, 2, axis=1) ** 2))
        print("Check 1...")
        print("Error for full sPOD recons: {}".format(err_full))

        ###########################################
        # Calculate the time amplitudes for training wildfire_data
        U_list = []
        spod_modes = []
        frame_amplitude_list_interpolation = []
        frame_amplitude_list_training = []
        cnt = 0
        for frame in sPOD_frames_train:
            VT = frame.modal_system["VT"]
            S = frame.modal_system["sigma"]
            VT = np.diag(S) @ VT
            Nmodes = frame.Nmodes
            amplitudes = [np.reshape(VT[n, :], [self.Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            frame_amplitude_list_interpolation.append(amplitudes)
            frame_amplitude_list_training.append(VT)
            U_list.append(frame.modal_system["U"])
            spod_modes.append(Nmodes)
            cnt = cnt + 1

        q_spod_frames = [sPOD_frames_train[0].build_field(),
                         sPOD_frames_train[1].build_field(),
                         sPOD_frames_train[2].build_field(),
                         qtilde_train]

        return q_spod_frames, U_list, frame_amplitude_list_training, frame_amplitude_list_interpolation, spod_modes

    def test_data(self, spod_iter):
        ##########################################
        # Calculate the transformation interpolation error
        dat = self.q_test
        data_shape = [self.Nx, 1, 1, self.Nt]
        dx = self.x[1] - self.x[0]
        L = [self.x[-1]]
        q = np.reshape(dat, data_shape)

        trafo_test_1 = transforms(data_shape, L, shifts=self.shifts_test[0], dx=[dx],
                                  use_scipy_transform=False,
                                  interp_order=5)
        trafo_test_2 = transforms(data_shape, L, shifts=self.shifts_test[1], trafo_type="identity",
                                  dx=[dx],
                                  use_scipy_transform=False, interp_order=5)
        trafo_test_3 = transforms(data_shape, L, shifts=self.shifts_test[2], dx=[dx],
                                  use_scipy_transform=False,
                                  interp_order=5)

        interp_err = give_interpolation_error(q, trafo_test_1)
        print("Transformation interpolation error =  %4.4e " % interp_err)

        ##########################################
        # run sPOD
        trafos_test = [trafo_test_1, trafo_test_2, trafo_test_3]

        qmat = np.reshape(self.q_test, [-1, self.Nt])
        [N, M] = np.shape(qmat)
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.001
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 5

        ret_test = shifted_rPCA(self.q_test, trafos_test, nmodes_max=5, eps=1e-16, Niter=spod_iter, use_rSVD=True,
                                mu=mu0, lambd=lambd0, dtol=1e-5)
        sPOD_frames_test, qtilde_test, rel_err_test = ret_test.frames, ret_test.data_approx, ret_test.rel_err_hist

        q1_test = sPOD_frames_test[0].build_field()
        q2_test = sPOD_frames_test[1].build_field()
        q3_test = sPOD_frames_test[2].build_field()

        Q_frames_test = [q1_test, q2_test, q3_test, qtilde_test]

        return Q_frames_test

    def plot_sPOD_frames(self, Q_frames_test):
        q1_spod_frame = Q_frames_test[0]
        q2_spod_frame = Q_frames_test[1]
        q3_spod_frame = Q_frames_test[2]
        qtilde_test = Q_frames_test[3]

        data_shape = [self.Nx, 1, 1, self.Nt]
        dx = self.x[1] - self.x[0]
        L = [self.x[-1]]
        trafo_test_1 = transforms(data_shape, L, shifts=self.shifts_test[0], dx=[dx],
                                  use_scipy_transform=False,
                                  interp_order=5)
        trafo_test_2 = transforms(data_shape, L, shifts=self.shifts_test[1], trafo_type="identity",
                                  dx=[dx],
                                  use_scipy_transform=False, interp_order=5)
        trafo_test_3 = transforms(data_shape, L, shifts=self.shifts_test[2], dx=[dx],
                                  use_scipy_transform=False,
                                  interp_order=5)

        q1_spod_frame = trafo_test_1.apply(q1_spod_frame)
        q2_spod_frame = trafo_test_2.apply(q2_spod_frame)
        q3_spod_frame = trafo_test_3.apply(q3_spod_frame)

        plot_sPODframes(self.q_test, q1_spod_frame, q2_spod_frame, q3_spod_frame, qtilde_test, self.x, self.t)

    def plot_online_data(self, frame_amplitude_predicted_sPOD, frame_amplitude_predicted_POD,
                         TA_TEST, TA_POD_TEST, TA_list_interp, shifts_predicted,
                         SHIFTS_TEST, spod_modes, U_list, U_POD_TRAIN, Q_frames_test,
                         plot_online=False, test_type=None):

        if test_type['typeOfTest'] == "query":
            plot_online = False
            test_sample = test_type['test_sample']
            self.q_test = self.q_test[:, test_sample][..., np.newaxis]
            self.shifts_train = np.asarray([self.shifts_train[:, i * self.Nt + test_sample]
                                            for i in range(self.Nsamples_train)]).transpose()

        print("#############################################")
        print('Online Error checks')
        # Online error with respect to testing data
        Nx = len(self.x)
        Nt = frame_amplitude_predicted_sPOD.shape[1]  # len(self.t)
        dx = self.x[1] - self.x[0]
        Nmf = spod_modes
        time_amplitudes_1_pred = frame_amplitude_predicted_sPOD[:Nmf[0], :]
        time_amplitudes_2_pred = frame_amplitude_predicted_sPOD[Nmf[0]:Nmf[0] + Nmf[1], :]
        time_amplitudes_3_pred = frame_amplitude_predicted_sPOD[Nmf[0] + Nmf[1]:, :]
        shifts_1_pred = shifts_predicted[0, :]
        shifts_3_pred = shifts_predicted[1, :]

        # Implement the interpolation to find the online prediction
        tic_I = time.process_time()
        shifts_list_interpolated = []
        cnt = 0
        for frame in range(self.NumFrames):
            shifts = np.reshape(self.shifts_train[cnt], [self.Nsamples_train, Nt]).T
            shifts_list_interpolated.append(shifts)
            cnt = cnt + 1

        DELTA_PRED_FRAME_WISE = my_delta_interpolate(shifts_list_interpolated, self.mu_vecs_train, self.mu_vecs_test)
        data_shape = [Nx, 1, 1, Nt]
        L = [self.x[-1]]
        tic_trafo_1 = time.process_time()
        trafo_interpolated_1 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[0], dx=[dx],
                                          use_scipy_transform=False,
                                          interp_order=5)
        trafo_interpolated_2 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[1], trafo_type="identity",
                                          dx=[dx],
                                          use_scipy_transform=False, interp_order=5)
        trafo_interpolated_3 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[2], dx=[dx],
                                          use_scipy_transform=False,
                                          interp_order=5)
        toc_trafo_1 = time.process_time()
        trafos_interpolated = [trafo_interpolated_1, trafo_interpolated_2, trafo_interpolated_3]

        QTILDE_FRAME_WISE, TA_INTERPOLATED = my_interpolated_state(spod_modes, U_list,
                                                                   TA_list_interp, self.mu_vecs_train,
                                                                   Nx, 1, Nt, self.mu_vecs_test,
                                                                   trafos_interpolated)
        toc_I = time.process_time()

        # Shifts error
        num1_i = np.linalg.norm(SHIFTS_TEST[0] - DELTA_PRED_FRAME_WISE[0])
        den1_i = np.linalg.norm(SHIFTS_TEST[0])
        num3_i = np.linalg.norm(SHIFTS_TEST[1] - DELTA_PRED_FRAME_WISE[2])
        den3_i = np.linalg.norm(SHIFTS_TEST[1])

        num1 = np.linalg.norm(SHIFTS_TEST[0] - shifts_1_pred.flatten())
        den1 = np.linalg.norm(SHIFTS_TEST[0])
        num3 = np.linalg.norm(SHIFTS_TEST[1] - shifts_3_pred.flatten())
        den3 = np.linalg.norm(SHIFTS_TEST[1])
        print('Check 1...')
        print("Relative error indicator for shift for frame 1 (sPOD-NN): {}".format(num1 / den1))
        print("Relative error indicator for shift for frame 3 (sPOD-NN): {}".format(num3 / den3))
        print("Relative error indicator for shift for frame 1 (sPOD-I): {}".format(num1_i / den1_i))
        print("Relative error indicator for shift for frame 3 (sPOD-I): {}".format(num3_i / den3_i))

        # Time amplitudes error
        time_amplitudes_1_test = TA_TEST[:Nmf[0], :]
        time_amplitudes_2_test = TA_TEST[Nmf[0]:Nmf[0] + Nmf[1], :]
        time_amplitudes_3_test = TA_TEST[Nmf[0] + Nmf[1]:, :]
        num1 = np.linalg.norm(time_amplitudes_1_test - time_amplitudes_1_pred)
        den1 = np.linalg.norm(time_amplitudes_1_test)
        num2 = np.linalg.norm(time_amplitudes_2_test - time_amplitudes_2_pred)
        den2 = np.linalg.norm(time_amplitudes_2_test)
        num3 = np.linalg.norm(time_amplitudes_3_test - time_amplitudes_3_pred)
        den3 = np.linalg.norm(time_amplitudes_3_test)

        num4 = np.linalg.norm(np.squeeze(time_amplitudes_1_test) - TA_INTERPOLATED[0])
        den4 = np.linalg.norm(np.squeeze(time_amplitudes_1_test))
        num5 = np.linalg.norm(np.squeeze(time_amplitudes_2_test) - TA_INTERPOLATED[1])
        den5 = np.linalg.norm(np.squeeze(time_amplitudes_2_test))
        num6 = np.linalg.norm(np.squeeze(time_amplitudes_3_test) - TA_INTERPOLATED[2])
        den6 = np.linalg.norm(np.squeeze(time_amplitudes_3_test))

        num7 = np.linalg.norm(TA_POD_TEST - frame_amplitude_predicted_POD)
        den7 = np.linalg.norm(TA_POD_TEST)
        print('Check 2...')
        print("Relative time amplitude error indicator for frame 1 (sPOD-NN): {}".format(num1 / den1))
        print("Relative time amplitude error indicator for frame 2 (sPOD-NN): {}".format(num2 / den2))
        print("Relative time amplitude error indicator for frame 3 (sPOD-NN): {}".format(num3 / den3))
        print("Relative time amplitude error indicator for frame 1 (sPOD-I): {}".format(num4 / den4))
        print("Relative time amplitude error indicator for frame 2 (sPOD-I): {}".format(num5 / den5))
        print("Relative time amplitude error indicator for frame 3 (sPOD-I): {}".format(num6 / den6))
        print("Relative time amplitude error indicator (POD-NN) is {}".format(num7 / den7))

        # Frame wise error
        tic_sPOD = time.process_time()
        q1_pred = U_list[0] @ time_amplitudes_1_pred
        q2_pred = U_list[1] @ time_amplitudes_2_pred
        q3_pred = U_list[2] @ time_amplitudes_3_pred

        Q_recon_sPOD = 0
        NumFrames = 3
        Q_pred = [np.reshape(q1_pred, newshape=data_shape),
                  np.reshape(q2_pred, newshape=data_shape),
                  np.reshape(q3_pred, newshape=data_shape)]

        shifts_1 = shifts_1_pred
        shifts_2 = np.zeros_like(shifts_1_pred)
        shifts_3 = shifts_3_pred
        L = [self.x[-1]]
        tic_trafo_2 = time.process_time()
        trafos_1 = transforms(data_shape, L, shifts=shifts_1, dx=[dx],
                              use_scipy_transform=False,
                              interp_order=5)
        trafos_2 = transforms(data_shape, L, shifts=shifts_2, trafo_type="identity", dx=[dx],
                              use_scipy_transform=False,
                              interp_order=5)
        trafos_3 = transforms(data_shape, L, shifts=shifts_3, dx=[dx],
                              use_scipy_transform=False,
                              interp_order=5)
        toc_trafo_2 = time.process_time()
        trafos = [trafos_1, trafos_2, trafos_3]
        for frame in range(NumFrames):
            Q_recon_sPOD += trafos[frame].apply(Q_pred[frame])
        toc_sPOD = time.process_time()

        tic_POD = time.process_time()
        Q_recon_POD = U_POD_TRAIN @ frame_amplitude_predicted_POD
        toc_POD = time.process_time()

        self.q_test = np.squeeze(self.q_test)
        Q_recon_sPOD = np.squeeze(Q_recon_sPOD)
        Q_recon_POD = np.squeeze(Q_recon_POD)
        QTILDE_FRAME_WISE = np.squeeze(QTILDE_FRAME_WISE)

        num1 = np.linalg.norm(self.q_test - Q_recon_sPOD)
        den1 = np.linalg.norm(self.q_test)

        num2 = np.linalg.norm(self.q_test - Q_recon_POD)
        den2 = np.linalg.norm(self.q_test)

        num1_i = np.linalg.norm(self.q_test - QTILDE_FRAME_WISE)
        den1_i = np.linalg.norm(self.q_test)

        print('Check 3...')
        print("Relative reconstruction error indicator for full snapshot (sPOD-NN): {}".format(num1 / den1))
        print("Relative reconstruction error indicator for full snapshot (sPOD-I): {}".format(num1_i / den1_i))
        print("Relative reconstruction error indicator for full snapshot (POD-NN): {}".format(num2 / den2))

        if test_type['typeOfTest'] != "query":
            one = self.q_test - Q_recon_sPOD
            num1 = np.sqrt(np.einsum('ij,ij->j', one, one))
            den1 = np.sqrt(np.sum(np.einsum('ij,ij->j', self.q_test, self.q_test)) / self.Nt)

            two = self.q_test - Q_recon_POD
            num2 = np.sqrt(np.einsum('ij,ij->j', two, two))

            three = self.q_test - QTILDE_FRAME_WISE
            num3 = np.sqrt(np.einsum('ij,ij->j', three, three))

            rel_err_sPOD = num1 / den1
            rel_err_POD = num2 / den1
            rel_err_interp = num3 / den1

            errors = [rel_err_sPOD, rel_err_POD, rel_err_interp]
        else:
            errors = [np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt)]
            
        if plot_online:
            if test_type['typeOfTest'] != "query":
                plot_pred_comb(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                               time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                               TA_INTERPOLATED, shifts_1_pred, shifts_3_pred, SHIFTS_TEST, DELTA_PRED_FRAME_WISE,
                               frame_amplitude_predicted_POD, TA_POD_TEST, self.x, self.t)
                plot_recons_snapshot_cross_section(self.q_test, QTILDE_FRAME_WISE, Q_recon_sPOD, Q_recon_POD, self.x,
                                                   self.t)

        print('Timing...')
        print(
            f"Time consumption in assembling the transformation operators (sPOD-NN) : {toc_trafo_2 - tic_trafo_2:0.4f} seconds")
        print(
            f"Time consumption in assembling the transformation operators (sPOD-I) : {toc_trafo_1 - tic_trafo_1:0.4f} seconds")
        print(f"Time consumption in assembling the final solution (sPOD-NN) : {((toc_sPOD - tic_sPOD) - (toc_trafo_2 - tic_trafo_2)):0.4f} seconds")
        print(f"Time consumption in assembling the final solution (sPOD-I)  : {((toc_I - tic_I) - (toc_trafo_1 - tic_trafo_1)):0.4f} seconds")
        print(f"Time consumption in assembling the final solution (POD-NN)  : {toc_POD - tic_POD:0.4f} seconds")

        return errors


def plot_sPODframes(q_test, q1_spod_frame, q2_spod_frame, q3_spod_frame, qtilde, x, t):
    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(qtilde)
    qmax = np.max(qtilde)
    fig, axs = plt.subplots(1, 4, num=3, sharey=True, figsize=(12, 5))
    bottom, top = 0.15, 0.9
    left, right = 0.1, 0.85
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0)
    # Reconstruction
    im = axs[0].pcolormesh(Xgrid, Tgrid, qtilde, vmin=qmin, vmax=qmax, cmap=cmap)
    # axs[0].axis('off')
    axs[0].axis('scaled')
    axs[0].set_xlabel(r"$Q$")
    axs[0].set_yticks([], [])
    axs[0].set_xticks([], [])
    # 1. frame
    axs[1].pcolormesh(Xgrid, Tgrid, q1_spod_frame, vmin=qmin, vmax=qmax, cmap=cmap)
    # axs[1].axis('off')
    axs[1].axis('scaled')
    axs[1].set_xlabel(r"$T^{\Delta^1}Q^1$")
    axs[1].set_yticks([], [])
    axs[1].set_xticks([], [])
    # 2. frame
    axs[2].pcolormesh(Xgrid, Tgrid, q2_spod_frame, vmin=qmin, vmax=qmax, cmap=cmap)
    # axs[2].axis('off')
    axs[2].axis('scaled')
    axs[2].set_xlabel(r"$T^{\Delta^2}Q^2$")
    axs[2].set_yticks([], [])
    axs[2].set_xticks([], [])
    # 3. frame
    axs[3].pcolormesh(Xgrid, Tgrid, q3_spod_frame, vmin=qmin, vmax=qmax, cmap=cmap)
    # axs[3].axis('off')
    axs[3].axis('scaled')
    axs[3].set_xlabel(r"$T^{\Delta^3}Q^3$")
    axs[3].set_yticks([], [])
    axs[3].set_xticks([], [])

    cbar_ax = fig.add_axes([0.90, bottom, 0.01, top - bottom])
    fig.colorbar(im, cax=cbar_ax)

    fig.supylabel(r"time $t$")
    fig.supxlabel(r"space $x$")

    save_fig(filepath=impath + "frames_sPOD", figure=fig)


def plot_pred_comb(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                   time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                   TA_interpolated, shifts_1_pred, shifts_3_pred, SHIFTS_TEST, delta_pred_frame_wise,
                   POD_frame_amplitudes_predicted, TA_POD_TEST, x, t):
    Nx = len(x)
    Nt = len(t)
    # Time amplitudes for all frames
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    axs[0, 0].plot(t, time_amplitudes_1_test[0, :Nt], color="green", linestyle='-', label='actual')
    axs[0, 0].plot(t, time_amplitudes_1_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    axs[0, 0].plot(t, TA_interpolated[0][0, :Nt], color="blue", linestyle='--', label='sPOD-I')
    axs[0, 0].set_xticks([0, t[-1] / 2, t[-1]])
    axs[0, 0].set_ylabel(r"$a_i^{k}(t,\mu)$")
    # axs[0, 0].set_title(r'${mode}^{(' + str(1) + ')}$')
    axs[0, 0].set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    axs[0, 0].set_xlabel(r"(a)")
    axs[0, 0].grid()
    axs[0, 0].legend(loc='lower right')

    axs[0, 1].plot(t, time_amplitudes_2_test[0, :Nt], color="green", linestyle='-', label='actual')
    axs[0, 1].plot(t, time_amplitudes_2_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    axs[0, 1].plot(t, TA_interpolated[1][0, :Nt], color="blue", linestyle='--', label='sPOD-I')
    axs[0, 1].set_xticks([0, t[-1] / 2, t[-1]])
    # axs[0, 1].set_ylabel(r"$a_i^{k}(t,\mu)$")
    # axs[0, 1].set_title(r'${mode}^{(' + str(1) + ')}$')
    axs[0, 1].set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    axs[0, 1].set_xlabel(r"(b)")
    axs[0, 1].grid()
    axs[0, 1].legend(loc='upper right')

    axs[1, 0].plot(t, SHIFTS_TEST[0].flatten()[:Nt], color="green", linestyle='-', label='actual')
    axs[1, 0].plot(t, shifts_1_pred.flatten()[:Nt], color="red", linestyle='--', label='sPOD-NN')
    axs[1, 0].plot(t, delta_pred_frame_wise[0][:Nt], color="blue", linestyle='--', label='sPOD-I')
    axs[1, 0].set_xticks([0, t[-1] / 2, t[-1]])
    axs[1, 0].set_ylabel(r"shifts $\underline{\Delta}^k$")
    # axs[1, 0].set_title(r"$\Delta$")
    axs[1, 0].set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    axs[1, 0].set_xlabel(r"(c)")
    axs[1, 0].grid()
    axs[1, 0].legend(loc='lower right')

    axs[1, 1].plot(t, SHIFTS_TEST[1].flatten()[:Nt], color="green", linestyle='-', label='actual')
    axs[1, 1].plot(t, shifts_3_pred.flatten()[:Nt], color="red", linestyle='--', label='sPOD-NN')
    axs[1, 1].plot(t, delta_pred_frame_wise[2][:Nt], color="blue", linestyle='--', label='sPOD-I')
    axs[1, 1].set_xticks([0, t[-1] / 2, t[-1]])
    # axs[1, 1].set_ylabel(r"space $x$")
    # axs[1, 1].set_title(r"$\Delta$")
    axs[1, 1].set_xticklabels(["0s", r"$1000s$", r"$2000s$"])
    axs[1, 1].set_xlabel(r"(d)")
    axs[1, 1].grid()
    axs[1, 1].legend(loc='lower left')

    axs[2, 0].plot(t, TA_POD_TEST[0, :], color="green", linestyle='-', label='actual')
    axs[2, 0].plot(t, POD_frame_amplitudes_predicted[0, :], color="black", linestyle='--', label='POD-NN')
    axs[2, 0].set_xticks([0, t[-1] / 2, t[-1]])
    axs[2, 0].set_ylabel(r"$a_i(t,\mu)$")
    axs[2, 0].set_xticklabels(["0s", r"$1000s$", r"$2000s$"])
    axs[2, 0].legend(loc='lower right')
    axs[2, 0].set_xlabel(r"(e)")
    axs[2, 0].grid()

    axs[2, 1].plot(t, TA_POD_TEST[1, :], color="green", linestyle='-', label='actual')
    axs[2, 1].plot(t, POD_frame_amplitudes_predicted[1, :], color="black", linestyle='--', label='POD-NN')
    axs[2, 1].set_xticks([0, t[-1] / 2, t[-1]])
    axs[2, 1].set_xticklabels(["0s", r"$1000s$", r"$2000s$"])
    axs[2, 1].legend(loc='upper right')
    axs[2, 1].set_xlabel(r"(f)")
    axs[2, 1].grid()

    fig.supxlabel(r"time $t$")
    save_fig(filepath=impath + "all_comb_pred", figure=fig)
    fig.savefig(impath + "all_comb_pred" + ".pdf", format='pdf', dpi=200, transparent=True, bbox_inches="tight")


def plot_recons_snapshot_cross_section(q_test, Q_recon_interp, Q_recon_sPOD, Q_recon_POD, x, t):
    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(q_test)
    qmax = np.max(q_test)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)

    # Create top/bottom subfigs
    (subfig_t, subfig_b) = fig.subfigures(1, 2, hspace=0.05, wspace=0.1)

    # put 3 axis in the top subfigure
    gs_t = subfig_t.add_gridspec(nrows=4, ncols=1)
    ax1 = subfig_t.add_subplot(gs_t[0:4, 0])
    # Original
    ax1.pcolormesh(Xgrid, Tgrid, q_test, vmin=qmin, vmax=qmax, cmap=cmap)
    ax1.axhline(y=t[Nt // 40], linestyle='--', color='r', label=r"$t=35s$")
    ax1.axhline(y=t[85 * Nt // 100], linestyle='--', color='g', label=r"$t=1190s$")
    ax1.set_title(r"$Q$")
    ax1.set_yticks([], [])
    ax1.set_xticks([], [])
    ax1.legend(loc='upper center')

    subfig_t.supylabel(r"time $t$")
    subfig_t.supxlabel(r"space $x$")

    # put 2 axis in the bottom subfigure
    gs_b = subfig_b.add_gridspec(nrows=4, ncols=1)
    ax4 = subfig_b.add_subplot(gs_b[0:2, 0])
    ax5 = subfig_b.add_subplot(gs_b[2:4, 0], sharex=ax4)

    start = 450
    end = 2550
    x_trim = x[start:end]
    ax4.plot(x_trim, q_test[start:end, 85 * Nt // 100], color="green", linestyle='-', label='actual')
    ax4.plot(x_trim, Q_recon_sPOD[start:end, 85 * Nt // 100], color="red", linestyle='--', label='sPOD-NN')
    ax4.plot(x_trim, Q_recon_interp[start:end, 85 * Nt // 100], color="blue", linestyle='--', label='sPOD-I')
    ax4.plot(x_trim, Q_recon_POD[start:end, 85 * Nt // 100], color="black", linestyle='--', label='POD-NN')
    axin2 = ax4.inset_axes([0.35, 0.2, 0.35, 0.35])
    axin2.plot(x[680:880], q_test[680:880, 85 * Nt // 100], color="green", linestyle='-')
    axin2.plot(x[680:880], Q_recon_sPOD[680:880, 85 * Nt // 100], color="red", linestyle='--')
    axin2.plot(x[680:880], Q_recon_interp[680:880, 85 * Nt // 100], color="blue", linestyle='--')
    axin2.plot(x[680:880], Q_recon_POD[680:880, 85 * Nt // 100], color="black", linestyle='--')
    axin2.set_xlim(230, 280)
    axin2.set_ylim(-80, 60)
    axin2.set_xticks([], [])
    axin2.set_yticks([], [])
    ax4.indicate_inset_zoom(axin2)
    # axs[1].set_yticks([], [])
    # axs[1].set_xticks([], [])
    ax4.set_title(r"$t=1190s$")

    ax4.grid()
    ax4.legend(loc='upper center')

    ax5.plot(x_trim, q_test[start:end, Nt // 40], color="green", linestyle='-', label='actual')
    ax5.plot(x_trim, Q_recon_sPOD[start:end, Nt // 40], color="red", linestyle='--', label='sPOD-NN')
    ax5.plot(x_trim, Q_recon_interp[start:end, Nt // 40], color="blue", linestyle='--', label='sPOD-I')
    ax5.plot(x_trim, Q_recon_POD[start:end, Nt // 40], color="black", linestyle='--', label='POD-NN')
    axin = ax5.inset_axes([0.6, 0.2, 0.35, 0.35])
    axin.plot(x[1560:1710], q_test[1560:1710, Nt // 40], color="green", linestyle='-')
    axin.plot(x[1560:1710], Q_recon_sPOD[1560:1710, Nt // 40], color="red", linestyle='--')
    axin.plot(x[1560:1710], Q_recon_interp[1560:1710, Nt // 40], color="blue", linestyle='--')
    axin.plot(x[1560:1710], Q_recon_POD[1560:1710, Nt // 40], color="black", linestyle='--')
    axin.set_xlim(520, 570)
    axin.set_ylim(-100, 100)
    axin.set_xticks([], [])
    axin.set_yticks([], [])
    ax5.indicate_inset_zoom(axin)
    # axs[2].set_yticks([], [])
    # axs[2].set_xticks([], [])
    ax5.set_title(r"$t=35s$")
    ax5.grid()
    ax5.legend(loc='upper left')

    subfig_b.supylabel(r"$T$")
    subfig_b.supxlabel(r"space $x$")

    fig.savefig(impath + "T_x_cross_section" + ".png", dpi=300, transparent=True)
