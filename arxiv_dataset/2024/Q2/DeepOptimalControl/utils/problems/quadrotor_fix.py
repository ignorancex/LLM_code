import torch
import numpy as np
from problems.problem import ControlProblem

torch.pi = torch.acos(torch.zeros(1)).item() * 2


class ProblemSetup(ControlProblem):
    def __init__(self, space_scale, tT, device="cpu"):
        super().__init__()
        self.space = space_scale
        self.tT = tT
        self.device = device

        self.N_states = 12
        self.N_controls = 4

        self.m = 2.0
        self.g = np.array([[0.],
                           [0.],
                           [9.81]])
        self.J = np.array([[1.2416, 0., 0.],
                           [0., 1.2416, 0.],
                           [0., 0., 2.*1.2416]])
        self.Jinv = np.linalg.inv(self.J)
        self.A = np.array([[0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [1., 0., 0., 0.]])
        self.B = np.array([[0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        self.U_d = np.array([[9.81*self.m], [0.], [0.], [0.]])
        self.U_d_torch = torch.tensor(
            self.U_d.astype(np.float32)).to(self.device)
        self.t_weight = 1.
        self.Qp = 0.0 * np.eye(3)
        self.Qv = 0.0 * np.eye(3)
        self.Qeta = 0. * np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 0.]])
        self.Qw = 0. * np.eye(3)
        self.Qu = 1. * np.eye(4)
        self.QpT = self.Qp.T+self.Qp
        self.QvT = self.Qv.T+self.Qv
        self.QetaT = self.Qeta.T+self.Qeta
        self.QwT = self.Qw.T+self.Qw
        self.QuTinv = np.linalg.inv(self.Qu.T+self.Qu)
        self.Qpf = np.zeros((3, 3))
        self.Qvf = 0.0 * np.eye(3)
        self.Qetaf = 0.0 * np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]])
        self.Qwf = 0.0 * np.eye(3)

        self.QpfT = self.Qpf.T+self.Qpf
        self.QvfT = self.Qvf.T+self.Qvf
        self.QetafT = self.Qetaf.T+self.Qetaf
        self.QwfT = self.Qwf.T+self.Qwf

        # init domian
        self.p0_ub = np.array([[40.], [40.], [40.]])
        self.p0_lb = np.array([[-40.], [-40.], [20.]])
        self.v0_ub = np.array([[1.], [1.], [1.]])
        self.v0_lb = - self.v0_ub
        self.eta0_ub = np.array(
            [[2 * np.pi/8.], [2 * np.pi/8.], [8. * np.pi/8.]])
        self.eta0_lb = - self.eta0_ub
        self.w0_ub = np.array([[0.], [0.], [0.]])
        self.w0_lb = - self.w0_ub

        self.X0_ub = np.vstack(
            (self.p0_ub, self.v0_ub, self.eta0_ub, self.w0_ub))
        self.X0_lb = np.vstack(
            (self.p0_lb, self.v0_lb, self.eta0_lb, self.w0_lb))

        self.X0_ub *= 0.1 * self.space
        self.X0_lb *= 0.1 * self.space
        print(self.X0_lb)

        self.w1 = 5.
        self.w2 = 10.
        self.w3 = 25.
        self.w4 = 50.

    def p_ref(self, t, p):
        # The desired trajectory for (x, y, z)
        p_d = np.array([np.zeros(t.shape),
                        np.zeros(t.shape),
                        np.zeros(t.shape)])

        return(np.reshape(p_d, np.shape(p)))

    def eta_ref(self, t, eta):
        # The desired trajectory for yaw angle
        eta_d = np.array([np.zeros(t.shape),
                          np.zeros(t.shape),
                          np.zeros(t.shape)])

        return(np.reshape(eta_d, np.shape(eta)))

    def U_star(self, X_aug):
        """ Control as a function of the costate. """
        Av = X_aug[15:18]
        Aw = X_aug[21:24]
        fu = np.matmul(self.A.T, Av)/self.m
        tau_u = np.matmul(self.B.T, np.matmul(self.Jinv.T, Aw))
        U = self.U_d.repeat(X_aug.shape[1], axis=1) - \
            np.matmul(self.QuTinv, fu+tau_u)
        return (U)

    def make_U_NN(self, u_model, t, X):
        """ Makes TensorFlow graph of optimal control with NN value gradient. """
        U = self.U_d_torch.T + u_model(t, X)
        return U

    def make_bc(self, X0_in):

        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.N_states]
            XT = X_aug_T[:self.N_states]
            AT = X_aug_T[self.N_states:2*self.N_states]

            # Derivative of the terminal cost with respect to X(T); 2 is necessary.
            dFdXT = 2 * \
                np.concatenate(
                    (self.w1 * XT[:3], self.w2 * XT[3:6], self.w3 * XT[6:9], self.w4 * XT[9:12]))
            return np.concatenate((X0 - X0_in, AT - dFdXT))

        return bc

    def running_cost(self, X, U):
        # NOTE: No t_weight in running cost.
        return np.sum((U - self.U_d)**2, axis=0, keepdims=True)

    def terminal_cost(self, X):
        p = X[0:3]
        v = X[3:6]
        eta = X[6:9]
        w = X[9:12]
        loss = self.w1 * p**2 + self.w2 * v**2 + self.w3 * eta**2 + self.w4 * w**2
        return np.sum(loss, axis=0, keepdims=True)

    def dynamics(self, t, X, U_fun):
        U = U_fun(np.array([[t]]), X.reshape((-1, 1))).flatten()

        v = X[3:6]
        eta = X[6:9]
        w = X[9:12]

        sin_eta = np.sin(eta)
        cos_eta = np.cos(eta)

        sin_phi = sin_eta[0]
        cos_phi = cos_eta[0]

        sin_theta = sin_eta[1]
        cos_theta = cos_eta[1]
        tan_theta = np.tan(eta[1])

        sin_psi = sin_eta[2]
        cos_psi = cos_eta[2]

        K = np.array([[1., sin_phi*tan_theta, cos_phi*tan_theta],
                      [0., cos_phi, -sin_phi],
                      [0., sin_phi/cos_theta, cos_phi/cos_theta]])
        S = np.array([[0., w[2], -w[1]],
                      [-w[2], 0., w[0]],
                      [w[1], -w[0], 0.]])
        R = np.array([[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
                      [sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                       sin_phi*sin_theta*sin_psi + cos_phi*cos_psi,
                       sin_phi*cos_theta],
                      [cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                       cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                       cos_phi*cos_theta]])
        RT = np.array([[cos_theta*cos_psi,
                        sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                        cos_phi*sin_theta*cos_psi + sin_phi*sin_psi],
                       [cos_theta*sin_psi,
                        sin_phi*sin_theta*sin_psi + cos_phi*cos_psi,
                        cos_phi*sin_theta*sin_psi - sin_phi*cos_psi],
                       [-sin_theta,
                        sin_phi*cos_theta,
                        cos_phi*cos_theta]])

        dpdt = np.matmul(RT, v)
        dvdt = np.matmul(S, v) - np.matmul(R, self.g).flatten() + \
            np.matmul(self.A, U)/self.m
        detadt = np.matmul(K, w)
        dwdt = np.matmul(S, np.matmul(self.J, w)) + np.matmul(self.B, U)
        dwdt = np.matmul(self.Jinv, dwdt)

        return np.concatenate((dpdt, dvdt, detadt, dwdt))

    def aug_dynamics(self, t, X_aug, tf=None):
        """ Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP. """
        # Control as a function of the costate
        U = self.U_star(X_aug)

        Nt = t.shape[0]
        zeros = np.zeros_like(t)
        long_zeros = np.zeros(3*Nt)
        ones = np.ones_like(t)

        # State
        p = X_aug[:3]
        v = X_aug[3:6]
        eta = X_aug[6:9]
        w = X_aug[9:12]

        # Costate
        Ap = X_aug[12:15]
        Av = X_aug[15:18]
        Aeta = X_aug[18:21]
        Aw = X_aug[21:24]

        sin_eta = np.sin(eta)
        cos_eta = np.cos(eta)
        sin_phi = sin_eta[0]
        cos_phi = cos_eta[0]
        sin_theta = sin_eta[1]
        cos_theta = cos_eta[1]
        tan_theta = np.tan(eta[1])
        sin_psi = sin_eta[2]
        cos_psi = cos_eta[2]

        # State dynamics
        K = np.vstack((np.concatenate((ones, sin_phi*tan_theta, cos_phi*tan_theta)),
                       np.concatenate((zeros, cos_phi, -sin_phi)),
                       np.concatenate((zeros, sin_phi/cos_theta, cos_phi/cos_theta))))
        S = np.vstack((np.concatenate((zeros, w[2], -w[1])),
                       np.concatenate((-w[2], zeros, w[0])),
                       np.concatenate((w[1], -w[0], zeros))))
        R = np.vstack((np.concatenate((cos_theta*cos_psi,
                                       cos_theta*sin_psi,
                                       -sin_theta)),
                       np.concatenate((sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                                       sin_phi*sin_theta*sin_psi + cos_phi*cos_psi,
                                       sin_phi*cos_theta)),
                       np.concatenate((cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                       cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                                       cos_phi*cos_theta))))
        # Reorganizes matrices into sample-contiguous blocks
        idx = np.array([0, Nt, 2*Nt])
        idx = np.concatenate(tuple(idx + k for k in range(Nt))).tolist()

        K_T = K[:, idx]
        S_T = S[:, idx]
        R_T = R[:, idx]

        # Most matrices need to be organized
        K = K.T.reshape(K.shape)
        S = S.T.reshape(S.shape)
        R = R.T.reshape(R.shape)

        # Makes large matrices of copies of p, v, omega, w
        V = np.repeat(v, 3, axis=1)
        W = np.repeat(w, 3, axis=1)
        G = np.repeat(self.g, 3*Nt, axis=1)
        JW = np.repeat(np.matmul(self.J, w), 3, axis=1)

        # Matrix-vector multiplication for all samples
        dpdt = np.sum(R_T*V, axis=0).reshape((3, Nt), order='F')
        dvdt = np.sum(S*V, axis=0).reshape((3, Nt), order='F')
        dvdt -= np.sum(R*G, axis=0).reshape((3, Nt), order='F')
        dvdt += np.matmul(self.A, U)/self.m
        detadt = np.sum(K*W, axis=0).reshape((3, Nt), order='F')
        dwdt = np.sum(S*JW, axis=0).reshape((3, Nt),
                                            order='F') + np.matmul(self.B, U)
        dwdt = np.matmul(self.Jinv, dwdt)

        # Costate dynamics
        DR1 = np.vstack((long_zeros,
                         np.concatenate((cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                         cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                                         cos_phi*cos_theta)),
                         np.concatenate((-sin_phi*sin_theta*cos_psi + cos_phi*sin_psi,
                                         -sin_phi*sin_theta*sin_psi - cos_phi*cos_psi,
                                         -sin_phi*cos_theta))))
        DR2 = np.vstack((np.concatenate((-cos_psi*sin_theta,
                                         -sin_psi*sin_theta,
                                         -cos_theta)),
                         np.concatenate((sin_phi*cos_theta*cos_psi,
                                         sin_phi*cos_theta*sin_psi,
                                         -sin_phi*sin_theta)),
                         np.concatenate((cos_phi*cos_theta*cos_psi,
                                         cos_phi*cos_theta*sin_psi,
                                         -cos_phi*sin_theta))))
        DR3 = np.vstack((np.concatenate((-cos_theta*sin_psi,
                                         cos_theta*cos_psi,
                                         zeros)),
                         np.concatenate((-sin_phi*sin_theta*sin_psi - cos_phi*cos_psi,
                                         sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                                         zeros)),
                         np.concatenate((-cos_phi*sin_theta*sin_psi + sin_phi*cos_psi,
                                         cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                         zeros))))
        DK1 = np.vstack((np.concatenate((zeros, cos_phi*tan_theta, -sin_phi*tan_theta)),
                         np.concatenate((zeros, -sin_phi, -cos_phi)),
                         np.concatenate((zeros, cos_phi/cos_theta, -sin_phi/cos_theta))))
        DK2 = np.vstack((np.concatenate((zeros,
                                         sin_phi/cos_theta**2,
                                         cos_phi/cos_theta**2)),
                         long_zeros,
                         np.concatenate((zeros,
                                         sin_phi*tan_theta/cos_theta,
                                         cos_phi*tan_theta/cos_theta))))
        DS1 = np.vstack((long_zeros,
                         np.concatenate((zeros, zeros, ones)),
                         np.concatenate((zeros, -ones, zeros))))
        DS2 = np.vstack((np.concatenate((zeros, zeros, -ones)),
                         long_zeros,
                         np.concatenate((ones, zeros, zeros))))
        DS3 = np.vstack((np.concatenate((zeros, ones, zeros)),
                         np.concatenate((-ones, zeros, zeros)),
                         long_zeros))

        # Reorganizes matrices into sample-contiguous blocks
        DR1_T = DR1[:, idx]
        DR2_T = DR2[:, idx]
        DR3_T = DR3[:, idx]
        DR1 = DR1.T.reshape(DR1.shape)
        DR2 = DR2.T.reshape(DR2.shape)
        DR3 = DR3.T.reshape(DR3.shape)
        DK1 = DK1.T.reshape(DK1.shape)
        DK2 = DK2.T.reshape(DK2.shape)
        DS1 = DS1.T.reshape(DS1.shape)
        DS2 = DS2.T.reshape(DS2.shape)
        DS3 = DS3.T.reshape(DS3.shape)

        AwJinvT = np.matmul(self.Jinv.T, Aw)
        AwJinv = np.repeat(AwJinvT, 3, axis=1)
        AwJinvS = np.sum(S_T*AwJinv, axis=0).reshape((3, Nt), order='F')
        AP = np.repeat(Ap, 3, axis=1)
        AV = np.repeat(Av, 3, axis=1)
        AEta = np.repeat(Aeta, 3, axis=1)

        dApdt = - np.zeros((3, Nt))
        dAvdt = - np.zeros((3, Nt))
        dAvdt -= np.sum(R*AP, axis=0).reshape((3, Nt), order='F')
        dAvdt -= np.sum(S_T*AV, axis=0).reshape((3, Nt), order='F')
        dAetadt = - np.zeros((3, Nt))
        dAetadt[0] -= np.sum(Aeta * np.sum(DK1*W,
                                           axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[1] -= np.sum(Aeta * np.sum(DK2*W,
                                           axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[0] += np.sum(Av * np.sum(DR1*G,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[1] += np.sum(Av * np.sum(DR2*G,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[2] += np.sum(Av * np.sum(DR3*G,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[0] -= np.sum(Ap * np.sum(DR1_T*V,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[1] -= np.sum(Ap * np.sum(DR2_T*V,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAetadt[2] -= np.sum(Ap * np.sum(DR3_T*V,
                                         axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt = - np.sum(K_T*AEta, axis=0).reshape((3, Nt), order='F')
        dAwdt[0] -= np.sum(Av * np.sum(DS1*V,
                                       axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt[1] -= np.sum(Av * np.sum(DS2*V,
                                       axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt[2] -= np.sum(Av * np.sum(DS3*V,
                                       axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt[0] -= np.sum(AwJinvT * np.sum(DS1*JW,
                                            axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt[1] -= np.sum(AwJinvT * np.sum(DS2*JW,
                                            axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt[2] -= np.sum(AwJinvT * np.sum(DS3*JW,
                                            axis=0).reshape((3, Nt), order='F'), axis=0)
        dAwdt -= np.matmul(self.J.T, AwJinvS)

        output = np.vstack(
            (dpdt, dvdt, detadt, dwdt, dApdt, dAvdt, dAetadt, dAwdt))
        return output
