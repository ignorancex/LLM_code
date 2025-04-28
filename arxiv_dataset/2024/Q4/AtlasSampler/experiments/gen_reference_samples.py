import numpy as np
from atlassampler import util
import nuts

class ReferenceSamples():


    def __init__(self,
                 stanfile,
                 datafile,
                 model_directory,
                 n_chains=16, 
                 n_burnin=1000,
                 step_size=0.01,
                 target_accept=0.95, 
                 n_metric_adapt=1000, 
                 n_stepsize_adapt=1000, 
                 metric='diag', 
                 seed=123,
                 savefolder=None):

        self.analytic_models = ['normal', 'funnel', 'rosenbrock', 'multifunnel',
                                'corr_normal90', 'corr_normal95']
        self.model_directory = model_directory
        self.stanfile = stanfile
        self.datafile = datafile
        self.n_chains = n_chains
        #self.n_burnin = n_burnin
        self.n_burnin = n_burnin + n_metric_adapt + n_stepsize_adapt 
        self.step_size = step_size
        self.target_accept = target_accept
        self.n_metric_adapt = 0 #n_metric_adapt #cmdstanpy bug?
        self.n_stepsize_adapt = 0 #n_stepsize_adapt
        self.metric = metric
        self.seed = seed
        self.savefolder = savefolder

    def gen_analytic_samples(self):

        if self.exp == 'normal':
            ref_samples = np.random.normal(0, 1, self.n_samples*self.D).reshape(self.n_samples, self.D)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'ill_normal':
            m = np.linspace(1, self.D, self.D)/self.D**0.5
            ref_samples = np.array([np.random.normal(0, m) for _ in range(self.n_samples)])
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'funnel':
            log_scale = np.random.normal(0, 3, self.n_samples)
            latents = np.array([np.random.normal(0, np.exp(log_scale/2)) for _ in range(self.D-1)]).T
            log_scale = log_scale.reshape(-1, 1)
            ref_samples = np.concatenate([log_scale, latents], axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'rosenbrock':
            x = np.random.normal(1, 1, self.n_samples)
            y = np.array([np.random.normal(x**2, 0.1) for _ in range(self.D-1)]).T
            x = np.expand_dims(x, axis=1)
            ref_samples = np.concatenate([x, y], axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if 'corr_normal' in self.exp:
            r = float(self.exp.split('corr_normal')[1])/100
            D = self.D
            m = np.zeros((D, D))
            for i in range(D):
                for j in range(D):
                    m[i, j] = r**abs(i-j)
            ref_samples = np.random.multivariate_normal(np.zeros(D), m, self.n_samples)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'multifunnel':
            assert self.D % 10 == 0 # D must be a multiple of 10
            copies = self.D//10
            ref_samples = []
            for i in range(copies):
                log_scale = np.random.normal(0, 3, self.n_samples).reshape(-1, 1)
                ref_samples.append(log_scale)
            for i in range(copies):                
                latents = np.array([np.random.normal(0, np.exp(ref_samples[i][:, 0]/2)) for _ in range(9)]).T
                ref_samples.append(latents)
            ref_samples = np.concatenate(ref_samples, axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        return ref_samples


    def nuts_samples(self):

        print("Running nuts to generate reference samples.\nThis can take time")

        args = {}
        args['n_chains'] = self.n_chains
        args['n_samples'] = self.n_samples
        args['n_burnin'] = self.n_burnin
        args['seed'] = self.seed
        args['metric'] = self.metric
        args['step_size'] = self.step_size
        args['target_accept'] = self.target_accept
        args['n_metric_adapt'] = self.n_metric_adapt
        args['n_stepsize_adapt'] = self.n_stepsize_adapt

        args = util.Objectify(args)
        print(args)
        print(args.n_samples)
        samples = nuts.run_nuts(self.stanfile, self.datafile, args,
                                seed=args.seed, savefolder=self.savefolder, verbose=True)
        return samples


    def generate_samples(self, exp, D, chain_index=True, run_nuts=True):
        self.D = D
        self.exp = exp
        self.chain_index = chain_index
        if exp in self.analytic_models:
            self.n_samples = 1000000
            return self.gen_analytic_samples()
        else:
            if run_nuts:
                self.n_samples = 10000
                return self.nuts_samples()
            else:
                return None
