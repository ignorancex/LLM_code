import librosa
import numpy as np

from utils import feature_extractor as utils


class EMG:
    def __init__(self, audio, config):
        self.audio = audio
        self.dependencies = config["emg"]["dependencies"]
        self.frame_size = int(config["frame_size"])
        self.sampling_rate = int(config["sampling_rate"])
        self.number_of_bins = int(config["emg"]["number_of_bins"])
        self.is_raw_data = config["is_raw_data"]
        self.time_lag = int(config["emg"]["time_lag"])
        self.embedded_dimension = int(config["emg"]["embedded_dimension"])
        self.boundary_frequencies = list(config["emg"]["boundary_frequencies"])
        self.hfd_parameter = int(config["emg"]["hfd_parameter"])
        self.r = int(config["emg"]["r"])
        self.frames = int(np.ceil(len(self.audio.data) / self.frame_size))

    def __enter__(self):
        print ("Initializing emg calculation...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print ("Done with calculations...")

    def get_current_frame(self, index):
            return utils._get_frame_array(self.audio, index, self.frame_size)

    def compute_hurst(self):
        self.hurst = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            N = current_frame.size
            T = np.arange(1, N + 1)
            Y = np.cumsum(current_frame)
            Ave_T = Y / T
            S_T = np.zeros(N)
            R_T = np.zeros(N)
            for i in range(N):
                S_T[i] = np.std(current_frame[:i + 1])
                X_T = Y - T * Ave_T[i]
                R_T[i] = np.ptp(X_T[:i + 1])

            R_S = R_T / S_T
            R_S = np.log(R_S)[1:]
            n = np.log(T)[1:]
            A = np.column_stack((n, np.ones(n.size)))
            [m, c] = np.linalg.lstsq(A, R_S)[0]
            self.hurst.append(m)
        self.hurst = np.asarray(self.hurst)

    def get_hurst(self):
        return self.hurst

    def compute_embed_seq(self):
        self.embed_seq = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            shape = (current_frame.size - self.time_lag * (self.embedded_dimension - 1), self.embedded_dimension)
            strides = (current_frame.itemsize, self.time_lag * current_frame.itemsize)
            m = np.lib.stride_tricks.as_strided(current_frame, shape=shape, strides=strides)
            self.embed_seq.append(m)
        self.embed_seq = np.asarray(self.embed_seq)

    def get_embed_seq(self):
        return self.embed_seq

    def compute_bin_power(self):
        self.Power_Ratio = []
        self.Power = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            C = np.fft.fft(current_frame)
            C = abs(C)
            Power = np.zeros(len(self.boundary_frequencies) - 1)
            for Freq_Index in range(0, len(self.boundary_frequencies) - 1):
                Freq = float(self.boundary_frequencies[Freq_Index])
                Next_Freq = float(self.boundary_frequencies[Freq_Index + 1])
                Power[Freq_Index] = sum(
                    C[int(np.floor(Freq / self.sampling_rate * len(current_frame))):
                    int(np.floor(Next_Freq / self.sampling_rate * len(current_frame)))])
            self.Power.append(Power)
            self.Power_Ratio.append(Power / sum(Power))
        self.Power = np.asarray(self.Power)
        self.Power_Ratio = np.asarray(self.Power_Ratio)

    def get_bin_power(self):
        return self.Power

    def get_bin_power_ratio(self):
        return self.Power_Ratio

    def compute_pfd(self, D=None):
        self.pfd = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            if D is None:
                D = np.diff(current_frame)
                D = D.tolist()
            N_delta = 0  # number of sign changes in derivative of the signal
            for i in range(1, len(D)):
                if D[i] * D[i - 1] < 0:
                    N_delta += 1
            n = len(current_frame)
            m = np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
            self.pfd.append(m)

        if self.is_raw_data:
            self.pfd = np.asarray(self.pfd)
        else:
            self.pfd = np.asarray(self.pfd)[0]

    def get_pfd(self):
        return self.pfd

    def compute_hfd(self):
        self.hfd = []
        for v in range(0, self.frames):
            current_frame = self.get_current_frame(v)
            L = []
            x = []
            N = len(current_frame)
            for k in range(1, self.hfd_parameter):
                Lk = []
                for m in range(0, k):
                    Lmk = 0
                    for i in range(1, int(np.floor((N - m) / k))):
                        Lmk += abs(current_frame[m + i * k] - current_frame[m + i * k - k])
                    Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                    Lk.append(Lmk)
                L.append(np.log(np.mean(Lk)))
                x.append([np.log(float(1) / k), 1])

            (p, r1, r2, s) = np.linalg.lstsq(x, L)
            self.hfd.append(p[0])

        if self.is_raw_data:
            self.hfd = np.asarray(self.hfd)
        else:
            self.hfd = np.asarray(self.hfd)[0]

    def get_hfd(self):
        return self.hfd

    def compute_hjorth(self, D=None):
        self.hjorth = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            if D is None:
                D = np.diff(current_frame)
            np.concatenate(([current_frame[0]], D))
            D = np.array(D)
            n = len(current_frame)
            M2 = float(sum(D ** 2)) / n
            TP = sum(np.array(current_frame) ** 2)
            M4 = 0
            for i in range(1, len(D)):
                M4 += (D[i] - D[i - 1]) ** 2
            M4 = M4 / n
            m = np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)
            self.hjorth.append(m)

        if self.is_raw_data:
            self.hjorth = np.asarray(self.hjorth)
        else:
            self.hjorth = np.asarray(self.hjorth)[0]

    def get_hjorth(self):
        return self.hjorth

    def compute_spectral_entropy(self):
        self.spectral_entropy = []
        for k in range(0, self.frames):
            Power, Power_Ratio = self.get_bin_power()[k], self.get_bin_power_ratio()[k]
            Spectral_Entropy = 0
            for i in range(0, len(Power_Ratio) - 1):
                Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
            Spectral_Entropy /= np.log(len(Power_Ratio))
            m = -1 * Spectral_Entropy
            self.spectral_entropy.append(m)
        self.spectral_entropy = np.asarray(self.spectral_entropy)


    def get_spectral_entropy(self):
        return self.spectral_entropy

    def compute_svd_entropy(self, W=None):
        self.svd_entropy = []
        for k in range(0, self.frames):
            if W is None:
                Y = self.get_embed_seq()[k]
                W = np.linalg.svd(Y, compute_uv=0)
                W /= sum(W)  # normalize singular values
            m = -1 * sum(W * np.log(W))
            self.svd_entropy.append(m)

        if self.is_raw_data:
            self.svd_entropy = np.asarray(self.svd_entropy)
        else:
            self.svd_entropy = np.asarray(self.svd_entropy)[0]

    def get_svd_entropy(self):
        return self.svd_entropy

    def compute_fisher_info(self, W=None):
        self.fisher_info = []
        for k in range(0, self.frames):
            if W is None:
                Y = self.get_embed_seq()
                W = np.linalg.svd(Y, compute_uv=0)
                W /= sum(W)  # normalize singular values
            m = -1 * sum(W * np.log(W))
            self.fisher_info.append(m)

        if self.is_raw_data:
            self.fisher_info = np.asarray(self.fisher_info)
        else:
            self.fisher_info = np.asarray(self.fisher_info)[0]

    def get_fisher_info(self):
        return self.fisher_info

    def compute_ap_entropy(self):
        self.ap_entropy = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            N = len(current_frame)
            Em = self.get_embed_seq()[k]
            A = np.tile(Em, (len(Em), 1, 1))
            B = np.transpose(A, [1, 0, 2])
            D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
            InRange = np.max(D, axis=2) <= self.r
            Cm = InRange.mean(axis=0)
            Dp = np.abs(np.tile(current_frame[self.embedded_dimension:],
                                (N - self.embedded_dimension, 1)) - np.tile(current_frame[self.embedded_dimension:],
                                                                            (N - self.embedded_dimension, 1)).T)
            Cmp = np.logical_and(Dp <= self.r, InRange[:-1, :-1]).mean(axis=0)
            Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))
            m = (Phi_m - Phi_mp) / (N - self.embedded_dimension)
            self.ap_entropy.append(m)

        self.ap_entropy = np.asarray(self.ap_entropy)

    def get_ap_entropy(self):
        return self.ap_entropy

    def compute_samp_entropy(self):
        self.samp_entropy = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            N = len(current_frame)
            Em = self.get_embed_seq()[k]
            A = np.tile(Em, (len(Em), 1, 1))
            B = np.transpose(A, [1, 0, 2])
            D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
            InRange = np.max(D, axis=2) <= self.r
            np.fill_diagonal(InRange, 0)  # Don't count self-matches
            Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
            Dp = np.abs(np.tile(current_frame[self.embedded_dimension:], (N - self.embedded_dimension, 1))
                        - np.tile(current_frame[self.embedded_dimension:], (N - self.embedded_dimension, 1)).T)
            Cmp = np.logical_and(Dp <= self.r, InRange[:-1, :-1]).sum(axis=0)
            # Uncomment below for old (miscounted) version
            # InRange[np.triu_indices(len(InRange))] = 0
            # InRange = InRange[:-1,:-2]
            # Cm = InRange.sum(axis=0) #  Probability that random M-sequences are in range
            # Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)
            # Dp = Dp[:,:-1]
            # Cmp = np.logical_and(Dp <= R, InRange).sum(axis=0)

            # Avoid taking log(0)
            Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))
            self.samp_entropy.append(Samp_En)
        self.samp_entropy = np.asarray(self.samp_entropy)


    def get_samp_entropy(self):
        return self.samp_entropy

    def compute_dfa(self, Ave=None, L=None):
        self.dfa = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            if Ave is None:
                Ave = np.mean(current_frame)
            Y = np.cumsum(current_frame)
            Y -= Ave
            if L is None:
                L = np.floor(len(current_frame) * 1 /
                             (2 ** np.array(list(range(4, int(np.log2(len(current_frame))) - 4)))))
            F = np.zeros(len(L))
            for i in range(0, len(L)):
                n = int(L[i])  # for each box length L[i]
                if n == 0:
                    print("time series is too short while the box length is too big")
                    print("abort")
                    exit()
                for j in range(0, len(current_frame), n):  # for each box
                    if j + n < len(current_frame):
                        c = list(range(j, j + n))
                        # coordinates of time in the box
                        c = np.vstack([c, np.ones(n)]).T
                        # the value of data in the box
                        y = Y[j:j + n]
                        # add residue in this box
                        F[i] += np.linalg.lstsq(c, y)[1]
                F[i] /= ((len(current_frame) / n) * n)
            F = np.sqrt(F)
            Alpha = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]
            self.dfa.append(Alpha)

        if self.is_raw_data:
            self.dfa = np.asarray(self.dfa)
        else:
            self.dfa = np.asarray(self.dfa)[0]

    def get_dfa(self):
        return self.dfa

    def compute_permutation_entropy(self):
        self.permutation_entropy = []
        for k in range(0, self.frames):
            PeSeq = []
            Em = self.get_embed_seq()[k]
            for i in range(0, len(Em)):
                r = []
                z = []
                for j in range(0, len(Em[i])):
                    z.append(Em[i][j])
                for j in range(0, len(Em[i])):
                    z.sort()
                    r.append(z.index(Em[i][j]))
                    z[z.index(Em[i][j])] = -1
                PeSeq.append(r)
            RankMat = []
            while len(PeSeq) > 0:
                RankMat.append(PeSeq.count(PeSeq[0]))
                x = PeSeq[0]
                for j in range(0, PeSeq.count(PeSeq[0])):
                    PeSeq.pop(PeSeq.index(x))
            RankMat = np.array(RankMat)
            RankMat = np.true_divide(RankMat, RankMat.sum())
            EntropyMat = np.multiply(np.log2(RankMat), RankMat)
            PE = -1 * EntropyMat.sum()
            self.permutation_entropy.append(PE)

        if self.is_raw_data:
            self.permutation_entropy = np.asarray(self.permutation_entropy)
        else:
            self.permutation_entropy = np.asarray(self.permutation_entropy)[0]

    def get_permutation_entropy(self):
        return self.permutation_entropy

    def compute_information_based_similarity(self, y, n):
        self.information_based_similarity = []
        for v in range(0, self.frames):
            current_frame = self.get_current_frame(v)
            Wordlist = []
            Space = [[0, 0], [0, 1], [1, 0], [1, 1]]
            Sample = [0, 1]

            if (n == 1):
                Wordlist = Sample

            if (n == 2):
                Wordlist = Space

            elif (n > 1):
                Wordlist = Space
                Buff = []
                for k in range(0, n - 2):
                    Buff = []

                    for i in range(0, len(Wordlist)):
                        Buff.append(tuple(Wordlist[i]))
                    Buff = tuple(Buff)

                    Wordlist = []
                    for i in range(0, len(Buff)):
                        for j in range(0, len(Sample)):
                            Wordlist.append(list(Buff[i]))
                            Wordlist[len(Wordlist) - 1].append(Sample[j])
            Wordlist.sort()
            Input = [[], []]
            Input[0] = current_frame
            Input[1] = y

            SymbolicSeq = [[], []]
            for i in range(0, 2):
                Encoder = np.diff(Input[i])
                for j in range(0, len(Input[i]) - 1):
                    if (Encoder[j] > 0):
                        SymbolicSeq[i].append(1)
                    else:
                        SymbolicSeq[i].append(0)

            Wm = []
            # todo fix this and uncomment these lines
            # Wm.append(self.get_embed_seq(SymbolicSeq[0], 1, n).tolist())
            # Wm.append(embed_seq(SymbolicSeq[1], 1, n).tolist())

            Count = [[], []]
            for i in range(0, 2):
                for k in range(0, len(Wordlist)):
                    Count[i].append(Wm[i].count(Wordlist[k]))

            Prob = [[], []]
            for i in range(0, 2):
                Sigma = 0
                for j in range(0, len(Wordlist)):
                    Sigma += Count[i][j]
                for k in range(0, len(Wordlist)):
                    Prob[i].append(np.true_divide(Count[i][k], Sigma))

            Entropy = [[], []]
            for i in range(0, 2):
                for k in range(0, len(Wordlist)):
                    if (Prob[i][k] == 0):
                        Entropy[i].append(0)
                    else:
                        Entropy[i].append(Prob[i][k] * (np.log2(Prob[i][k])))

            Rank = [[], []]
            Buff = [[], []]
            Buff[0] = tuple(Count[0])
            Buff[1] = tuple(Count[1])
            for i in range(0, 2):
                Count[i].sort()
                Count[i].reverse()
                for k in range(0, len(Wordlist)):
                    Rank[i].append(Count[i].index(Buff[i][k]))
                    Count[i][Count[i].index(Buff[i][k])] = -1

            IBS = 0
            Z = 0
            n = 0
            for k in range(0, len(Wordlist)):
                if ((Buff[0][k] != 0) & (Buff[1][k] != 0)):
                    F = -Entropy[0][k] - Entropy[1][k]
                    IBS += np.multiply(np.absolute(Rank[0][k] - Rank[1][k]), F)
                    Z += F
                else:
                    n += 1

            IBS = np.true_divide(IBS, Z)
            IBS = np.true_divide(IBS, len(Wordlist) - n)
            self.information_based_similarity.append(IBS)

        if self.is_raw_data:
            self.information_based_similarity = np.asarray(self.information_based_similarity)
        else:
            self.information_based_similarity = np.asarray(self.information_based_similarity)[0]

    def get_information_based_similarity(self):
        return self.information_based_similarity

    def compute_LLE(self, T):
        self.LLE = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            Em = self.get_embed_seq()
            M = len(Em)
            A = np.tile(Em, (len(Em), 1, 1))
            B = np.transpose(A, [1, 0, 2])
            square_dists = (A - B) ** 2  # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
            D = np.sqrt(square_dists[:, :, :].sum(axis=2))  # D[i,j] = ||Em[i]-Em[j]||_2

            # Exclude elements within T of the diagonal
            band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
            band[band == 1] = np.inf
            neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

            # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
            inc = np.tile(np.arange(M), (M, 1))
            row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
            col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
            in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)
            # Uncomment for old (miscounted) version
            # in_bounds = np.logical_and(row_inds < M - 1, col_inds < M - 1)
            row_inds[-in_bounds] = 0
            col_inds[-in_bounds] = 0

            # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
            neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], -in_bounds)
            J = (-neighbor_dists.mask).sum(axis=1)  # number of in-bounds indices by row
            # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged
            neighbor_dists[neighbor_dists == 0] = 1
            d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
            mean_d = d_ij[J > 0] / J[J > 0]

            x = np.arange(len(mean_d))
            X = np.vstack((x, np.ones(len(mean_d)))).T
            [m, c] = np.linalg.lstsq(X, mean_d)[0]
            # todo check fs
            Lexp = self.sampling_rate * m
            self.LLE.append(Lexp)

        if self.is_raw_data:
            self.LLE = np.asarray(self.LLE)
        else:
            self.LLE = np.asarray(self.LLE)[0]

    def get_LLE(self, tau, n, T):
        return self.LLE
