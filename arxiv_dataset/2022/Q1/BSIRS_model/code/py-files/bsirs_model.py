import numpy as np
from scipy.special import expit


class BSIRS:
    def __init__(self, w_matrix, pops, tau, infect):
        self.w_matrix = w_matrix
        self.length = w_matrix.shape[0]
        self.pops = pops
        self.tau = tau
        self.infected_original = infect
        self.infected = infect
        self.time = 0
        self.u_value = np.full(self.length, 0.5)
        self.susceptibles = self.pops - self.infected
        self.recovered = np.zeros(self.length, dtype="int32")

    def reset(self):
        """Resets the values of
        self.time,
        self.infected,
        self.suscepfibles,
        self.recovered
        ...
        :return: None
        :rtype: NoneType
        """
        self.time = 0
        self.infected = np.copy(self.infected_original)
        self.susceptibles = self.pops - self.infected
        self.recovered = np.zeros(self.length, dtype="int32")

    def change_infected(self, infected, susceptibles, u_vals):
        self.u_value = u_vals
        self.infected = infected
        self.susceptibles = susceptibles
        self.recovered = self.pops - infected - susceptibles

    def calculate_contact_matrix(self):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        h_value = 1.0 - self.u_value
        b_matrix = np.array(h_value * self.infected)[:, np.newaxis]
        infected_hat = h_value * (self.w_matrix @ b_matrix .reshape(-1))
        return infected_hat

    def logit_inv(self, beta, c_val, ubar):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        return expit(beta*(ubar - c_val))

    def update(
        self,
        beta,
        c_val,
        epsilon,
        prob_rec,
        prob_xi,
        random=False
    ):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        infected_hat = self.calculate_contact_matrix()
        lam_s = prob_xi
        lam_i = epsilon * infected_hat
        lam_r = prob_rec
        if random:
            xsi = np.random.binomial(self.susceptibles, p=lam_i/self.pops)
            xir = np.random.binomial(self.infected, p=lam_r)
            xrs = np.random.binomial(self.recovered, p=lam_s)
        else:
            xsi = (self.susceptibles * lam_i / self.pops).astype('int32')
            xir = (self.infected * lam_r).astype('int32')
            xrs = (self.recovered * lam_s).astype('int32')
        self.susceptibles = self.susceptibles + xrs - xsi
        self.infected = self.infected + xsi - xir
        self.recovered = self.recovered + xir - xrs
        ubar = self.w_matrix @ self.u_value
        probab = self.logit_inv(beta, c_val, ubar)
        self.u_value = (1 - 1/self.tau)*self.u_value + 1/self.tau*probab
        self.time += 1
