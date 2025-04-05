class SMDL:
    """
    Sequential Minimum Description Length algorithm
    """
    #def __init__(self, h, T, model, beta):
    #def __init__(self, model, beta):
    def __init__(self, model):
        """
        initialize parameters
        :param h: window size
        :param T: data length
        :param model: model instance (necessary to implement 'calc_change_score' method)
        :param beta: threshold parameter
        :return:
        """
        #self.h = h
        #self.T = T
        #self.model = model
        #self.model = model(h, T)
        #self.model = model()
        self.model = model
        #self.beta = beta

    #def calc_change_score(self, x, mu_max, sigma_min):
    #def calc_change_score(self, x, **kwargs):
    def calc_change_score(self, x, t, **kwargs):
        """
        calculate change score using specified model
        :param x: data
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change score
        """
        # calculate change score
        change_score = self.model.calc_change_score(x, t, **kwargs)
        
        return change_score
    
    def calc_change_score_1st(self, x, t, **kwargs):
        """
        calculate 1st differential MDL change statistics using specified model
        :param x: data
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change score
        """
        # calculate change score
        change_score = self.model.calc_change_score_1st(x, t, **kwargs)
        
        return change_score 
    
    
    def calc_change_score_2nd(self, x, t, **kwargs):
        """
        calculate 2nd differential MDL change statistics using specified model
        :param x: data
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change score
        """
        # calculate change score
        change_score = self.model.calc_change_score_2nd(x, t, **kwargs)

        return change_score