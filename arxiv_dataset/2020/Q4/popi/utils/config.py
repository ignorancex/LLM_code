class Config(object):
    def __init__(self, method_name, generations, pop_size, clip_min, clip_max, a, b, scale, p, std, gamma, CR,
                 best, dist_name, indices, patience):
        self.config = {}
        self.config['method_name'] = method_name

        self.config['generations'] = generations
        self.config['pop_size'] = pop_size
        # if self.config['method_name'] in ['ABC_DE', 'DE', 'RevDE']:
        #     self.config['generations'] = generations
        #     self.config['pop_size'] = pop_size
        # elif self.config['method_name'] in ['ABC_MH']:
        #     self.config['generations'] = generations * pop_size
        #     self.config['pop_size'] = 1

        self.config['clip_min'] = clip_min
        self.config['clip_max'] = clip_max

        self.config['a'] = a
        self.config['b'] = b
        self.config['scale'] = scale

        self.config['p'] = p

        self.config['std'] = std
        self.config['gamma'] = gamma
        self.config['CR'] = CR
        self.config['best'] = best

        self.config['dist_name'] = dist_name

        self.config['indices'] = indices

        self.config['patience'] = patience