import scipy.stats as stats


class GeneralMethod(object):
    def __init__(self, conf):
        self.conf = conf
        self.gamma = self.conf['gamma']
        self.CR = self.conf['CR']
        self.scale = self.conf['scale']
        self.a = self.conf['a']
        self.b = self.conf['b']
        self.best = self.conf['best']
        self.p = self.conf['p']
        self.std = self.conf['std']
        self.clip_min = self.conf['clip_min']
        self.clip_max = self.conf['clip_max']

        self.indices = self.conf['indices']

        self.dist_name = self.conf['dist_name']
        if self.dist_name == 'truncnorm':
            self.dist = stats.truncnorm
        elif self.dist_name == 'norm':
            self.dist = stats.norm
        else:
            self.dist = None

    def proposal(self, theta, E=None):
        pass

    def step(self, theta, E_old, x_obs, mod, params):
        pass