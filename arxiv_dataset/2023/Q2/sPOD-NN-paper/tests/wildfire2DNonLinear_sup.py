import time
from Helper import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from wildfire2D_sup import cartesian_to_polar, polar_to_cartesian

impath = "../plots/images_wildfire2DNonLinear/"
os.makedirs(impath, exist_ok=True)

data_path = os.path.abspath(".") + '/wildfire_data/2DNonLinear/'

cmap = 'YlOrRd'

# cmap = 'YlGn'


class wildfire2DNonLinear_sup:
    def __init__(self, q_test, shifts_test, param_test_val, var):
        dat1_train = np.load(data_path + 'SnapShotMatrix540.npy')
        dat2_train = np.load(data_path + 'SnapShotMatrix550.npy')
        dat3_train = np.load(data_path + 'SnapShotMatrix560.npy')
        dat4_train = np.load(data_path + 'SnapShotMatrix570.npy')
        dat5_train = np.load(data_path + 'SnapShotMatrix580.npy')
        self.grid_1D = np.load(data_path + '1D_Grid.npy', allow_pickle=True)
        self.grid_2D = np.load(data_path + '2D_Grid.npy', allow_pickle=True)
        self.x = self.grid_1D[0]
        self.y = self.grid_1D[1]
        self.X = self.grid_2D[0]
        self.Y = self.grid_2D[1]
        self.t = np.load(data_path + 'Time.npy')
        delta1_train = np.load(data_path + 'Shifts540.npy')
        delta2_train = np.load(data_path + 'Shifts550.npy')
        delta3_train = np.load(data_path + 'Shifts560.npy')
        delta4_train = np.load(data_path + 'Shifts570.npy')
        delta5_train = np.load(data_path + 'Shifts580.npy')

        self.truncate_shift_rank = 4

        self.var = var
        self.Nx = np.size(self.x)
        self.Ny = np.size(self.y)
        self.Nt = np.size(self.t)
        self.x_c = self.x[-1] // 2
        self.y_c = self.y[-1] // 2

        # Test data
        self.param_test_val = param_test_val
        self.mu_vecs_test = np.asarray([self.param_test_val])
        self.q_test = np.reshape(np.transpose(q_test), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        self.q_test = np.transpose(np.reshape(np.squeeze(self.q_test[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        self.shifts_test = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in shifts_test]
        self.q_polar_test = None

        # Train data
        self.Nsamples_train = 5
        self.NumFrames = 2
        self.mu_vecs_train = np.asarray([540, 550, 560, 570, 580])
        self.params_train = [np.squeeze(np.asarray([[np.ones_like(self.t) * mu], [self.t]])) for mu in
                             self.mu_vecs_train]
        self.params_train = np.concatenate(self.params_train, axis=1)

        delta1_train = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in delta1_train]
        delta2_train = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in delta2_train]
        delta3_train = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in delta3_train]
        delta4_train = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in delta4_train]
        delta5_train = [np.reshape(x, newshape=[2, -1, self.Nt]) for x in delta5_train]
        self.shifts_train = []
        self.shifts_train.append(np.concatenate((delta1_train[0], delta2_train[0],
                                                 delta3_train[0], delta4_train[0], delta5_train[0]), axis=-1))
        self.shifts_train.append(np.concatenate((delta1_train[1], delta2_train[1],
                                                 delta3_train[1], delta4_train[1], delta5_train[1]), axis=-1))

        # Extract relevant shift amplitudes from the 2D shifts
        self.shift_U_test, self.shift_TA_test = truncate_shifts(self.shifts_test)
        self.shift_U_train, self.shift_TA_train = truncate_shifts(self.shifts_train)

        dat1_train = np.reshape(np.transpose(dat1_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat1_train = np.transpose(np.reshape(np.squeeze(dat1_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat2_train = np.reshape(np.transpose(dat2_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat2_train = np.transpose(np.reshape(np.squeeze(dat2_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat3_train = np.reshape(np.transpose(dat3_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat3_train = np.transpose(np.reshape(np.squeeze(dat3_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat4_train = np.reshape(np.transpose(dat4_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat4_train = np.transpose(np.reshape(np.squeeze(dat4_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat5_train = np.reshape(np.transpose(dat5_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat5_train = np.transpose(np.reshape(np.squeeze(dat5_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        self.q_train = [dat1_train, dat2_train, dat3_train, dat4_train, dat5_train]
        self.q_polar_train = None

    def run_sPOD(self, spod_iter):
        # Reshape the variable array to suit the dimension of the input for the sPOD
        self.q_train = [np.reshape(q, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F") for q in self.q_train]

        # Map the field variable from cartesian to polar coordinate system
        q_polar = []
        s0, theta_i, r_i, _ = cartesian_to_polar(self.q_train[0], self.x, self.y, self.t)
        q_polar.append(s0)
        for samples in range(self.Nsamples_train - 1):
            s, _, _, _ = cartesian_to_polar(self.q_train[samples + 1], self.x, self.y, self.t)
            q_polar.append(s)

        data_shape = [self.Nx, self.Ny, 1, self.Nsamples_train * self.Nt]
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])

        # Create the transformations
        trafo_train_1 = transforms(data_shape, L, shifts=self.shifts_train[0],
                                   dx=d_del,
                                   use_scipy_transform=False)
        trafo_train_2 = transforms(data_shape, L, shifts=self.shifts_train[1],
                                   trafo_type="identity", dx=d_del,
                                   use_scipy_transform=False)

        # Apply srPCA on the data
        transform_list = [trafo_train_1, trafo_train_2]

        qmat = np.concatenate([np.reshape(q, newshape=[-1, self.Nt]) for q in q_polar], axis=1)
        mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.1
        lambd = 1 / np.sqrt(np.max([self.Nx, self.Ny])) * 5.0

        ret = shifted_rPCA(qmat, transform_list, nmodes_max=10, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu,
                           lambd=lambd)
        sPOD_frames_train, qtilde_train, rel_err_train = ret.frames, ret.data_approx, ret.rel_err_hist
        self.q_polar_train = qmat
        ###########################################
        # Calculate the time amplitudes for training data
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

        return U_list, frame_amplitude_list_training, frame_amplitude_list_interpolation, spod_modes

    def test_data(self, spod_iter):
        ##########################################
        # Reshape the variable array to suit the dimension of the input for the sPOD
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")

        # Map the field variable from cartesian to polar coordinate system
        q_polar, theta_i, r_i, aux = cartesian_to_polar(q, self.x, self.y, self.t, fill_val=0)

        # Check the transformation back and forth error between polar and cartesian coordinates (Checkpoint)
        q_cartesian = polar_to_cartesian(q_polar, self.t, aux=aux)
        res = q - q_cartesian
        err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
        print("Transformation back and forth error (cartesian - polar - cartesian) =  %4.4e " % err)

        data_shape = [self.Nx, self.Ny, 1, self.Nt]
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])

        trafo_test_1 = transforms(data_shape, L, shifts=self.shifts_test[0],
                                  dx=d_del,
                                  use_scipy_transform=False)
        trafo_test_2 = transforms(data_shape, L, shifts=self.shifts_test[1],
                                  trafo_type="identity", dx=d_del,
                                  use_scipy_transform=False)

        # Check the transformation interpolation error
        err = give_interpolation_error(q_polar, trafo_test_1)
        print("Transformation interpolation error =  %4.4e " % err)

        ##########################################
        # Apply sPOD on the data
        transform_list = [trafo_test_1, trafo_test_2]
        qmat = np.reshape(q_polar, [-1, self.Nt])
        mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.5
        lambd = 1 / np.sqrt(np.max([self.Nx, self.Ny])) * 5.0
        ret = shifted_rPCA(qmat, transform_list, nmodes_max=10, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu,
                           lambd=lambd)
        sPOD_frames_test, qtilde_test, rel_err_test = ret.frames, ret.data_approx, ret.rel_err_hist
        self.q_polar_test = qmat

        # Deduce the frames
        modes_list = [sPOD_frames_test[0].Nmodes, sPOD_frames_test[1].Nmodes]
        q_frame_1 = sPOD_frames_test[0].build_field()
        q_frame_2 = sPOD_frames_test[1].build_field()
        qtilde = np.reshape(qtilde_test, newshape=data_shape)

        # Transform the frame wise snapshots into lab frame (moving frame)
        q_frame_1_lab = transform_list[0].apply(np.reshape(q_frame_1, newshape=data_shape))
        q_frame_2_lab = transform_list[1].apply(np.reshape(q_frame_2, newshape=data_shape))

        # Shift the pre-transformed polar data to cartesian grid to visualize
        q_frame_1_cart_lab = polar_to_cartesian(q_frame_1_lab, self.t, aux=aux)
        q_frame_2_cart_lab = polar_to_cartesian(q_frame_2_lab, self.t, aux=aux)
        qtilde_cart = polar_to_cartesian(qtilde, self.t, aux=aux)

        # Relative reconstruction error for sPOD
        res = q - qtilde_cart
        err_full = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
        print("Error for full sPOD recons: {}".format(err_full))

        Q_frames_test_polar = [q_frame_1, q_frame_2, qtilde_test]
        Q_frames_test_cart = [q_frame_1_cart_lab, q_frame_2_cart_lab, qtilde_cart]

        return Q_frames_test_polar, Q_frames_test_cart, aux


    def plot_sPOD_frames(self, Q_frames_test_cart, plot_every=10, var_name="T"):
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")
        qframe0_lab = Q_frames_test_cart[0]
        qframe1_lab = Q_frames_test_cart[1]
        qtilde = Q_frames_test_cart[2]

        immpath = impath + "srPCA_2D/"
        os.makedirs(immpath, exist_ok=True)
        for n in range(self.Nt):
            if n % plot_every == 0:
                min = np.min(q[..., 0, n])
                max = np.max(q[..., 0, n])

                fig = plt.figure(figsize=(12, 5), constrained_layout=True)
                (subfig_t) = fig.subfigures(1, 1, hspace=0.05, wspace=0.1)

                gs_t = subfig_t.add_gridspec(nrows=1, ncols=6)
                ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
                ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1, sharey=ax1)
                ax3 = subfig_t.add_subplot(gs_t[0, 4:6], sharex=ax1, sharey=ax1)

                ax1.pcolormesh(self.X, self.Y, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax1.axis('scaled')
                ax1.set_title(r"sPOD")
                ax1.set_yticks([], [])
                ax1.set_xticks([], [])

                ax2.pcolormesh(self.X, self.Y, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax2.axis('scaled')
                ax2.set_title(r"Frame $1$")
                ax2.set_yticks([], [])
                ax2.set_xticks([], [])

                ax3.pcolormesh(self.X, self.Y, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax3.axis('scaled')
                ax3.set_title(r"Frame $2$")
                ax3.set_yticks([], [])
                ax3.set_xticks([], [])

                subfig_t.supylabel(r"space $y$")
                subfig_t.supxlabel(r"space $x$")

                fig.savefig(immpath + str(var_name) + "-" + str(n), dpi=200, transparent=True)
                plt.close(fig)

    def plot_online_data(self, frame_amplitude_predicted_sPOD, frame_amplitude_predicted_POD,
                         TA_TEST, TA_POD_TEST, TA_list_interp, shifts_predicted, SHIFTS_TEST, spod_modes,
                         U_list, U_POD_TRAIN, q_test_polar, Q_frames_test_polar, aux, plot_online=False):

        Ndims = 2
        Nt = frame_amplitude_predicted_sPOD.shape[1]
        Q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")
        q_test_polar = np.reshape(q_test_polar, newshape=[self.Nx, self.Ny, 1, self.Nt])
        print("#############################################")
        print('Online Error checks')
        # %% Online error with respect to testing wildfire_data
        Nmf = spod_modes
        time_amplitudes_1_pred = frame_amplitude_predicted_sPOD[:Nmf[0], :]
        time_amplitudes_2_pred = frame_amplitude_predicted_sPOD[Nmf[0]:, :]
        shift_TA_pred = shifts_predicted

        X_new = self.X - self.x_c  # Shift the origin to the center of the image
        Y_new = self.Y - self.y_c
        r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
        theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta
        r_i = np.linspace(np.min(r), np.max(r), self.Nx)
        theta_i = np.linspace(np.min(theta), np.max(theta), self.Ny)
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])
        data_shape = [self.Nx, self.Ny, 1, Nt]

        # Implement the interpolation to find the online prediction
        tic_I = time.process_time()
        shifts_TA_list_interpolated = []
        for rank in range(self.truncate_shift_rank):
            shifts_TA_list_interpolated.append(
                np.reshape(self.shift_TA_train[rank], [self.Nsamples_train, Nt]).T)
        DELTA_TA = my_delta_interpolate(shifts_TA_list_interpolated, self.mu_vecs_train, self.mu_vecs_test)
        DELTA_TA = [x[np.newaxis, ...] for x in DELTA_TA]
        DELTA_TA = np.concatenate(DELTA_TA, axis=0)

        DELTA_PRED_FRAME_WISE = [np.zeros_like(self.shifts_test[0]), np.zeros_like(self.shifts_test[1])]
        DELTA_PRED_FRAME_WISE[0][0] = self.shift_U_train @ DELTA_TA
        DELTA_PRED_FRAME_WISE[0][1] = 0
        DELTA_PRED_FRAME_WISE[1][0] = 0
        DELTA_PRED_FRAME_WISE[1][1] = 0

        tic_trafo_1 = time.process_time()
        trafo_interpolated_1 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[0],
                                          dx=d_del,
                                          use_scipy_transform=False)
        trafo_interpolated_2 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[1],
                                          trafo_type="identity", dx=d_del,
                                          use_scipy_transform=False)
        toc_trafo_1 = time.process_time()
        trafos_interpolated = [trafo_interpolated_1, trafo_interpolated_2]
        QTILDE_FRAME_WISE, TA_INTERPOLATED = my_interpolated_state(spod_modes, U_list,
                                                                   TA_list_interp, self.mu_vecs_train,
                                                                   self.Nx, self.Ny, Nt, self.mu_vecs_test,
                                                                   trafos_interpolated)
        toc_I = time.process_time()

        # Shifts error
        num1 = np.linalg.norm(self.shifts_test[0][0] - self.shift_U_train @ shift_TA_pred)
        den1 = np.linalg.norm(self.shifts_test[0][0])
        num1_i = np.linalg.norm(self.shifts_test[0][0] - DELTA_PRED_FRAME_WISE[0][0])
        den1_i = np.linalg.norm(self.shifts_test[0][0])
        print('Check 1...')
        print("Relative error indicator for shift for frame 1 (sPOD-NN): {}".format(num1 / den1))
        print("Relative error indicator for shift for frame 1 (sPOD-I): {}".format(num1_i / den1_i))

        # Time amplitudes error
        time_amplitudes_1_test = TA_TEST[:Nmf[0], :]
        time_amplitudes_2_test = TA_TEST[Nmf[0]:, :]
        num1 = np.linalg.norm(time_amplitudes_1_test - time_amplitudes_1_pred)
        den1 = np.linalg.norm(time_amplitudes_1_test)
        num2 = np.linalg.norm(time_amplitudes_2_test - time_amplitudes_2_pred)
        den2 = np.linalg.norm(time_amplitudes_2_test)
        num1_i = np.linalg.norm(np.squeeze(time_amplitudes_1_test) - TA_INTERPOLATED[0])
        den1_i = np.linalg.norm(np.squeeze(time_amplitudes_1_test))
        num2_i = np.linalg.norm(np.squeeze(time_amplitudes_2_test) - TA_INTERPOLATED[1])
        den2_i = np.linalg.norm(np.squeeze(time_amplitudes_2_test))
        num3 = np.linalg.norm(TA_POD_TEST - frame_amplitude_predicted_POD)
        den3 = np.linalg.norm(TA_POD_TEST)
        print('Check 2...')
        print("Relative time amplitude error indicator (polar) for frame 1 (sPOD-NN): {}".format(num1 / den1))
        print("Relative time amplitude error indicator (polar) for frame 2 (sPOD-NN): {}".format(num2 / den2))
        print("Relative time amplitude error indicator (polar) for frame 1 (sPOD-I): {}".format(num1_i / den1_i))
        print("Relative time amplitude error indicator (polar) for frame 2 (sPOD-I): {}".format(num2_i / den2_i))
        print("Relative time amplitude error indicator (polar) (POD-NN): {}".format(num3 / den3))

        tic_sPOD = time.process_time()
        q1_pred = U_list[0] @ time_amplitudes_1_pred
        q2_pred = U_list[1] @ time_amplitudes_2_pred
        NumFrames = 2
        data_shape = [self.Nx, self.Ny, 1, Nt]
        Q_pred = [np.reshape(q1_pred, newshape=data_shape), np.reshape(q2_pred, newshape=data_shape)]
        Q_recon_sPOD_polar = np.zeros_like(q_test_polar)

        shifts = [np.zeros_like(self.shifts_test[0]), np.zeros_like(self.shifts_test[1])]
        shifts[0][0] = self.shift_U_train @ shift_TA_pred
        shifts[0][1] = 0
        shifts[1][0] = 0
        shifts[1][1] = 0

        tic_trafo_2 = time.process_time()
        trafos_1 = transforms(data_shape, L, shifts=shifts[0], dx=d_del, use_scipy_transform=False)
        trafos_2 = transforms(data_shape, L, shifts=shifts[1], trafo_type="identity", dx=d_del, use_scipy_transform=False)
        toc_trafo_2 = time.process_time()
        trafos = [trafos_1, trafos_2]
        for frame in range(NumFrames):
            Q_recon_sPOD_polar += trafos[frame].apply(Q_pred[frame])
        toc_sPOD = time.process_time()

        res = np.squeeze(np.reshape(q_test_polar - Q_recon_sPOD_polar, newshape=[-1, 1, self.Nt], order="F"))
        err_full_sPOD = np.linalg.norm(res) / np.linalg.norm(np.squeeze(np.reshape(q_test_polar, newshape=[-1, 1, self.Nt], order="F")))

        res = np.squeeze(np.reshape(q_test_polar - QTILDE_FRAME_WISE, newshape=[-1, 1, self.Nt], order="F"))
        err_full_interp = np.linalg.norm(res) / np.linalg.norm(np.squeeze(np.reshape(q_test_polar, newshape=[-1, 1, self.Nt], order="F")))

        print('Check 3...')
        print("Relative reconstruction error indicator for full snapshot (polar) (sPOD-NN): {}".format(err_full_sPOD))
        print(
            "Relative reconstruction error indicator for full snapshot (polar) (sPOD-I): {}".format(err_full_interp))

        # Convert the polar data into cartesian data
        tic_sPOD_cart = time.process_time()
        Q_recon_sPOD_cart = polar_to_cartesian(Q_recon_sPOD_polar, self.t, aux=aux)
        toc_sPOD_cart = time.process_time()

        tic_POD = time.process_time()
        Q_recon_POD_cart = U_POD_TRAIN @ frame_amplitude_predicted_POD
        Q_recon_POD_cart = np.reshape(Q_recon_POD_cart, newshape=data_shape, order="F")
        toc_POD = time.process_time()

        tic_I_cart = time.process_time()
        Q_recon_interp_cart = polar_to_cartesian(QTILDE_FRAME_WISE, self.t, aux=aux)
        toc_I_cart = time.process_time()

        res = np.squeeze(np.reshape(Q - Q_recon_sPOD_cart, newshape=[-1, 1, self.Nt], order="F"))
        err_full_sPOD = np.linalg.norm(res) / np.linalg.norm(self.q_test)
        res = np.squeeze(np.reshape(Q - Q_recon_POD_cart, newshape=[-1, 1, self.Nt], order="F"))
        err_full_POD = np.linalg.norm(res) / np.linalg.norm(self.q_test)
        res = np.squeeze(np.reshape(Q - Q_recon_interp_cart, newshape=[-1, 1, self.Nt], order="F"))
        err_full_interp = np.linalg.norm(res) / np.linalg.norm(self.q_test)
        print('Check 4...')
        print("Relative reconstruction error indicator for full snapshot (cartesian) (sPOD-NN): {}".format(
            err_full_sPOD))
        print("Relative reconstruction error indicator for full snapshot (cartesian) (sPOD-I): {}".format(
            err_full_interp))
        print("Relative reconstruction error indicator for full snapshot (cartesian) (POD-NN): {}".format(err_full_POD))

        Q_diff_sPOD = np.concatenate([Q[:, :, 0, n].flatten('F') - Q_recon_sPOD_cart[:, :, 0, n].flatten('F')
                                      for n in range(Nt)]).reshape(self.Nx * self.Ny, Nt, order='F')
        Q_diff_POD = np.concatenate([Q[:, :, 0, n].flatten('F') - Q_recon_POD_cart[:, :, 0, n].flatten('F')
                                     for n in range(Nt)]).reshape(self.Nx * self.Ny, Nt, order='F')
        Q_diff_interp = np.concatenate([Q[:, :, 0, n].flatten('F') - Q_recon_interp_cart[:, :, 0, n].flatten('F')
                                        for n in range(Nt)]).reshape(self.Nx * self.Ny, Nt, order='F')
        Q_act = np.concatenate([Q[:, :, 0, n].flatten('F')
                                for n in range(Nt)]).reshape(self.Nx * self.Ny, Nt, order='F')
        num1 = np.sqrt(np.einsum('ij,ij->j', Q_diff_sPOD, Q_diff_sPOD))
        den1 = np.sqrt(np.sum(np.einsum('ij,ij->j', Q_act, Q_act)) / self.Nt)
        num2 = np.sqrt(np.einsum('ij,ij->j', Q_diff_POD, Q_diff_POD))
        num3 = np.sqrt(np.einsum('ij,ij->j', Q_diff_interp, Q_diff_interp))

        rel_err_sPOD_cart = num1 / den1
        rel_err_POD_cart = num2 / den1
        rel_err_interp_cart = num3 / den1

        errors = [rel_err_sPOD_cart, rel_err_POD_cart, rel_err_interp_cart]

        print('Timing...')
        print(
            f"Time consumption in assembling the transformation operators (sPOD-NN) : {toc_trafo_2 - tic_trafo_2:0.4f} seconds")
        print(
            f"Time consumption in assembling the transformation operators (sPOD-I) : {toc_trafo_1 - tic_trafo_1:0.4f} seconds")
        print(
            f"Time consumption in assembling the final solution (sPOD-NN) : {((toc_sPOD - tic_sPOD) - (toc_trafo_2 - tic_trafo_2)):0.4f} seconds")
        print(
            f"Time consumption in assembling the final solution (sPOD-I)  : {((toc_I - tic_I) - (toc_trafo_1 - tic_trafo_1)):0.4f} seconds")
        print(f"Time consumption in assembling the final solution (POD-NN)  : {(toc_POD - tic_POD):0.4f} seconds")
        print(
            f"Time consumption in converting from cart-polar-cart  : {(2 * (toc_sPOD_cart - tic_sPOD_cart)):0.4f} seconds")

        return Q_recon_sPOD_cart, Q_recon_POD_cart, Q_recon_interp_cart, errors

    def plot_recon(self, Q_recon_sPOD_cart, Q_recon_POD_cart, Q_recon_interp_cart, t_a=10, t_b=100, var_name="T"):
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")

        min_a = np.min(q[..., 0, t_a - 1])
        max_a = np.max(q[..., 0, t_a - 1])

        min_b = np.min(q[..., 0, t_b - 1])
        max_b = np.max(q[..., 0, t_b - 1])

        fig = plt.figure(figsize=(11, 11), constrained_layout=True)
        (subfig_t, subfig_b) = fig.subfigures(2, 1, hspace=0.05, wspace=0.1)

        gs_t = subfig_t.add_gridspec(nrows=1, ncols=4)
        ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
        ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1, sharey=ax1)

        ax1.pcolormesh(self.X, self.Y, np.squeeze(q[:, :, 0, t_a - 1]), vmin=min_a, vmax=max_a, cmap=cmap)
        ax1.axis('scaled')
        ax1.set_title(r"$t=50s$")
        ax1.set_yticks([], [])
        ax1.set_xticks([], [])
        ax1.axhline(y=self.y[self.Ny // 2 - 1], linestyle='--', color='g')
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', size='5%', pad=0.08)
        # fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2.pcolormesh(self.X, self.Y, np.squeeze(q[:, :, 0, t_b - 1]), vmin=min_b, vmax=max_b, cmap=cmap)
        ax2.axis('scaled')
        ax2.set_title(r"$t=500s$")
        ax2.set_yticks([], [])
        ax2.set_xticks([], [])
        ax2.axhline(y=self.y[self.Ny // 2 - 1], linestyle='--', color='g')
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes('right', size='5%', pad=0.08)
        # fig.colorbar(im2, cax=cax, orientation='vertical')

        subfig_t.supylabel(r"space $y$")
        subfig_t.supxlabel(r"space $x$")

        gs_b = subfig_b.add_gridspec(nrows=1, ncols=4)
        ax4 = subfig_b.add_subplot(gs_b[0, 0:2])
        ax5 = subfig_b.add_subplot(gs_b[0, 2:4], sharex=ax4)

        ax4.plot(self.x, np.squeeze(q[:, self.Ny // 2, 0, t_a - 1]), color="green", linestyle="-", label='actual')
        ax4.plot(self.x, np.squeeze(Q_recon_sPOD_cart[:, self.Ny // 2, 0, t_a - 1]), color="red", linestyle="--",
                 label='sPOD-NN')
        ax4.plot(self.x, np.squeeze(Q_recon_interp_cart[:, self.Ny // 2, 0, t_a - 1]), color="blue", linestyle="-.",
                 label='sPOD-I')
        ax4.plot(self.x, np.squeeze(Q_recon_POD_cart[:, self.Ny // 2, 0, t_a - 1]), color="black", linestyle="-.",
                 label='POD-NN')
        ax4.set_ylim(bottom=min_a - 100, top=max_a + 300)
        ax4.legend()
        ax4.grid()

        ax5.plot(self.x, np.squeeze(q[:, self.Ny // 2, 0, t_b - 1]), color="green", linestyle="-", label='actual')
        ax5.plot(self.x, np.squeeze(Q_recon_sPOD_cart[:, self.Ny // 2, 0, t_b - 1]), color="red", linestyle="--",
                 label='sPOD-NN')
        ax5.plot(self.x, np.squeeze(Q_recon_interp_cart[:, self.Ny // 2, 0, t_b - 1]), color="blue", linestyle="-.",
                 label='sPOD-I')
        ax5.plot(self.x, np.squeeze(Q_recon_POD_cart[:, self.Ny // 2, 0, t_b - 1]), color="black", linestyle="-.",
                 label='POD-NN')
        ax5.set_ylim(bottom=min_b - 100, top=max_b + 300)
        ax5.legend()
        ax5.grid()

        subfig_b.supylabel(r"$T$")
        subfig_b.supxlabel(r"space $x$")

        fig.savefig(impath + str(var_name) + "-mixed", dpi=300, transparent=True)
        plt.close(fig)


def plot_pred_comb(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                   time_amplitudes_2_test, TA_interpolated, shifts_1_pred, SHIFTS_TEST,
                   shifts_interp, POD_frame_amplitudes_predicted, TA_POD_TEST, x, t):
    Nx = len(x)
    Nt = len(t)

    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    (subfig_t) = fig.subfigures(1, 1, hspace=0.05, wspace=0.1)

    gs_t = subfig_t.add_gridspec(nrows=3, ncols=4)
    ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
    ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1)
    ax3 = subfig_t.add_subplot(gs_t[1, 0:2], sharex=ax1)
    ax4 = subfig_t.add_subplot(gs_t[1, 2:4], sharex=ax1)
    ax5 = subfig_t.add_subplot(gs_t[2, 1:3], sharex=ax1)

    ax1.plot(t, time_amplitudes_1_test[0, :Nt], color="green", linestyle='-', label='actual')
    ax1.plot(t, time_amplitudes_1_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    ax1.plot(t, TA_interpolated[0][0, :Nt], color="blue", linestyle='--', label='sPOD-I')
    ax1.set_xticks([0, t[-1] / 2, t[-1]])
    ax1.set_ylabel(r"$a_i^{k}(t,\mu)$")
    ax1.set_xticklabels([r"$0s$", r"$250s$", r"$500s$"])
    ax1.set_xlabel(r"(a)")
    ax1.grid()
    ax1.legend(loc='lower right')

    ax2.plot(t, time_amplitudes_2_test[0, :Nt], color="green", linestyle='-', label='actual')
    ax2.plot(t, time_amplitudes_2_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    ax2.plot(t, TA_interpolated[1][0, :Nt], color="blue", linestyle='--', label='sPOD-I')
    ax2.set_xticks([0, t[-1] / 2, t[-1]])
    ax2.set_xticklabels([r"$0s$", r"$250s$", r"$500s$"])
    ax2.set_xlabel(r"(b)")
    ax2.grid()
    ax2.legend(loc='upper right')

    ax3.plot(t, TA_POD_TEST[0, :], color="green", linestyle='-', label='actual')
    ax3.plot(t, POD_frame_amplitudes_predicted[0, :], color="black", linestyle='--', label='POD-NN')
    ax3.set_xticks([0, t[-1] / 2, t[-1]])
    ax3.set_ylabel(r"$a_i^{k}(t,\mu)$")
    ax3.set_xticklabels(["0s", r"$250s$", r"$500s$"])
    ax3.set_xlabel(r"(c)")
    ax3.legend(loc='upper left')
    ax3.grid()

    ax4.plot(t, TA_POD_TEST[1, :], color="green", linestyle='-', label='actual')
    ax4.plot(t, POD_frame_amplitudes_predicted[1, :], color="black", linestyle='--', label='POD-NN')
    ax4.set_xticks([0, t[-1] / 2, t[-1]])
    ax4.set_xticklabels(["0s", r"$250s$", r"$500s$"])
    ax4.set_xlabel(r"(d)")
    ax4.legend(loc='lower left')
    ax4.grid()

    ax5.plot(t, SHIFTS_TEST[0, :Nt], color="green", linestyle='-', label='actual')
    ax5.plot(t, shifts_1_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    ax5.plot(t, shifts_interp[0, :Nt], color="blue", linestyle='--', label='sPOD-I')
    ax5.set_xticks([0, t[-1] / 2, t[-1]])
    ax5.set_xticklabels([r"$0s$", r"$250s$", r"$500s$"])
    ax5.set_ylabel(r"space $x$")
    ax5.set_xlabel(r"(e)")
    ax5.grid()
    ax5.legend(loc='upper right')

    subfig_t.supxlabel(r"time $t$")

    save_fig(filepath=impath + "all_comb_pred", figure=fig)
    fig.savefig(impath + "all_comb_pred" + ".eps", format='eps', dpi=600, transparent=True)



def truncate_shifts(delta, rank=4):

    # Shifts considered here are for the frame 1 and r coordinate. Rest all the shifts are zero by problem design
    shifts = np.squeeze(delta[0][0, ...])
    U, S, VT = np.linalg.svd(shifts, full_matrices=False)

    TA = np.diag(S[:rank]).dot(VT[:rank, :])

    return U[:, :rank], TA
