#mergers_w_humans_tools.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import logsumexp
import pymc3 as pm
import theano.tensor as TT
from scipy.special import beta as Beta

plt.ion()

def make_sample(N, f_M, r_Ms, r_Is, seed):
    seed = 1234
    np.random.seed(seed)
    true_gals = np.random.choice([0,1], size=N, p=[1-f_M, f_M])
    f_M_sample = true_gals.sum()/N
    N_true = true_gals.sum()

    n = len(r_Ms)


    # Matrix of classifier answers
    m = np.zeros((n, N), dtype='int')
    for i in range(n):
        for j in range(N):
            if true_gals[j] == 0:
                m[i,j] = np.random.choice([0,1], p=[r_Is[i], 1-r_Is[i]])
            elif true_gals[j] == 1:
                m[i,j] = np.random.choice([0,1], p=[1-r_Ms[i], r_Ms[i]])


    return N_true, m, true_gals

def make_flat(observations):
    n_cfers, n_obj = observations.shape
    obs_flat = observations.ravel()

    cfer_ids = np.zeros([n_cfers, n_obj], dtype='int')
    for i in range(n_cfers):
        cfer_ids[i] = i
    cfer_ids = cfer_ids.ravel()

    obj_ids = np.zeros([n_cfers, n_obj], dtype='int')
    for j in range(n_obj):
        obj_ids[:, j] = j
    obj_ids = obj_ids.ravel()

    return obs_flat, obj_ids, cfer_ids

def get_beta_pars(truths, observed):
    t0 = (truths==0)
    t1 = (truths==1)
    o0 = (observed==0)
    o1 = (observed==1)

    a_00 = np.count_nonzero(t0 & o0, axis=1) + 1
    b_00 = np.count_nonzero(t0 & o1, axis=1) + 1
    b_11 = np.count_nonzero(t1 & o0, axis=1) + 1
    a_11 = np.count_nonzero(t1 & o1, axis=1) + 1

    return a_00, b_00, a_11, b_11

def get_chains(obs, obs_train, true_samp,draws):
    '''
    these arrays are n_galaxies x n_classifiers, and are 0 for isolated, 1 for merger.
    '''
    obs_classes, obj_ids, cfer_ids = make_flat(obs)
    a_00, b_00, a_11, b_11 = get_beta_pars(true_samp, obs_train)

    n_cfers, n_obj = obs.shape

    with pm.Model() as explicit_model:
        f_M  = pm.Beta('fM', alpha=0.5, beta=0.5)
        r_Is = pm.Beta('rI', alpha=a_00, beta=b_00, shape=n_cfers)
        r_Ms = pm.Beta('rM', alpha=a_11, beta=b_11, shape=n_cfers)

        #the "real" data
        true_class = pm.Bernoulli('true_class', f_M, shape=n_obj)
        N_M_true = pm.Deterministic('N_M', TT.sum(true_class))
        p_obs_1 = TT.switch(TT.eq(true_class[obj_ids], 1), r_Ms[cfer_ids], 1-r_Is[cfer_ids])
        observed = pm.Bernoulli('observed', p_obs_1, observed=obs_classes)

    with explicit_model:
        trace = pm.sample(draws=draws)

    return trace.get_values('fM'), trace.get_values('rI'), trace.get_values('rM'), trace.get_values('true_class')

def make_obs(n_cfers, f_M_true, f_M_train, n_obj, n_train):


    seed = 0

    np.random.seed(seed)
    cfer_probs = np.random.uniform(low=0.5, high=0.9, size=n_cfers*2).reshape(-1,2)

    r_Is = cfer_probs[:, 0]
    r_Ms = cfer_probs[:, 1]


    true, obs, true_samp = make_sample(int(n_obj), f_M_true, r_Ms, r_Is, seed=1234)

    true_train, obs_train, true_samp_train = make_sample(int(n_train), f_M_train, r_Ms, r_Is, seed=123)
    return obs, true, obs_train, true_train, true_samp_train

def analyze_sim_sample(obs, obs_train, true_train, true_samp, with_plots,flat_prior):

    f_Ms, r_Is, r_Ms, classif = get_chains(obs, obs_train, true_train)

    if with_plots == 'y':

        plt.figure()

        plt.hist(f_Ms, bins=np.linspace(0,1,101), density=True)
        plt.xlabel(r'$f_M$')
        print(np.percentile(f_Ms, [16, 50, 84]))
        # Binomial distribution assuming that all of the galaxies were classified perfectly
        # Fundamental limit to which a sample's merger fraction can be determined.
        alpha=int(np.median(f_Ms)*n_obj)+1
        beta = n_obj-true+1
        p = np.linspace(0, 0.5, 1000)
        plt.plot(p,p**(alpha-1)*(1-p)**(beta-1)/Beta(alpha,beta))


    n_cfers, n_obj = obs.shape

    if flat_prior == 'n':

        p_Ms = np.zeros((1000, n_obj))
        p_Ms = []
        for k in np.random.choice(np.arange(len(f_Ms)), size=1000):
            for j in range(n_obj):
                num = 1
                den_I = 1
                den_M = 1
                for i in range(n_cfers):
                    if obs[i,j] == 0:
                        num *= (1-r_Ms[k,i])
                        den_M *= (1-r_Ms[k,i])
                        den_I *= r_Is[k,i]
                    else:
                        num *= r_Ms[k,i]
                        den_M *= r_Ms[k,i]
                        den_I *= (1-r_Ms[k,i])
                p_Ms.append(num*f_Ms[k]/(den_M*f_Ms[k]+den_I*(1-f_Ms[k])))
        p_Ms = np.array(p_Ms)
        p_Ms = p_Ms.reshape(-1, n_obj)

        if with_plots == 'y':
            plt.figure(figsize=(12,4))
            yll, mu, yul = np.percentile(p_Ms, [16,50,68], axis=0)
            inds = (true_samp == 0)
            plt.errorbar(np.arange(n_obj)[inds], mu[inds], yerr=[mu[inds]-yll[inds], yul[inds]-mu[inds]], fmt='.')
            inds = (true_samp == 1)
            plt.errorbar(np.arange(n_obj)[inds], mu[inds], yerr=[mu[inds]-yll[inds], yul[inds]-mu[inds]], fmt='.')
            plt.xlabel('Galaxy index')
            plt.ylabel(r'$p_M$')




    '''
    with flat prior
    '''

    if flat_prior == 'y':

        p_Ms = np.zeros((1000, n_obj))
        p_Ms = []
        for k in np.random.choice(np.arange(len(f_Ms)), size=1000):
            for j in range(n_obj):
                num = 1
                den_I = 1
                den_M = 1
                for i in range(n_cfers):
                    if obs[i,j] == 0:
                        num *= (1-r_Ms[k,i])
                        den_M *= (1-r_Ms[k,i])
                        den_I *= r_Is[k,i]
                    else:
                        num *= r_Ms[k,i]
                        den_M *= r_Ms[k,i]
                        den_I *= (1-r_Ms[k,i])
                p_Ms.append(num/(den_M+den_I))
        p_Ms = np.array(p_Ms)
        p_Ms = p_Ms.reshape(-1, n_obj)

        if with_plots == 'y':
            plt.figure(figsize=(12,4))
            yll, mu, yul = np.percentile(p_Ms, [16,50,68], axis=0)
            inds = (true_samp == 0)
            plt.errorbar(np.arange(n_obj)[inds], mu[inds], yerr=[mu[inds]-yll[inds], yul[inds]-mu[inds]], fmt='.')
            inds = (true_samp == 1)
            plt.errorbar(np.arange(n_obj)[inds], mu[inds], yerr=[mu[inds]-yll[inds], yul[inds]-mu[inds]], fmt='.')
            plt.xlabel('Galaxy index')
            plt.ylabel(r'$p_M$')
            plt.title('Flat prior')

    return f_Ms, r_Is, r_Ms, classif, p_Ms

def analyze_sample(obs, obs_train, true_samp, with_plots,draws):

    f_Ms, r_Is, r_Ms, classif = get_chains(obs, obs_train, true_samp,draws)

    if with_plots == 'y':

        plt.figure()

        plt.hist(f_Ms, bins=np.linspace(0,1,101), density=True)
        plt.ylabel(r'Probability')
        plt.xlabel(r'Merger Fraction ($f_M$)')
        plt.title('Merger Fraction Distribution of a Sample')
        print(r"Median Merger Fraction: " + str(round(np.percentile(f_Ms, [16, 50, 84])[1],2)) + r'$\pm$' + str(round(f_Ms.std(),2)))
        # Binomial distribution assuming that all of the galaxies were classified perfectly
        # Fundamental limit to which a sample's merger fraction can be determined.
        #alpha=int(np.median(f_Ms)*n_obj)+1
        #beta = n_obj-true+1
        #p = np.linspace(0, 0.5, 1000)
        #plt.plot(p,p**(alpha-1)*(1-p)**(beta-1)/Beta(alpha,beta))

    n_cfers, n_obj = obs.shape

    p_Ms = np.zeros((1000, n_obj))
    p_Ms = []
    for k in np.random.choice(np.arange(len(f_Ms)), size=1000):
        for j in range(n_obj):
            num = 1
            den_I = 1
            den_M = 1
            for i in range(n_cfers):
                if obs[i,j] == 0:
                    num *= (1-r_Ms[k,i])
                    den_M *= (1-r_Ms[k,i])
                    den_I *= r_Is[k,i]
                else:
                    num *= r_Ms[k,i]
                    den_M *= r_Ms[k,i]
                    den_I *= (1-r_Ms[k,i])
            p_Ms.append(num*f_Ms[k]/(den_M*f_Ms[k]+den_I*(1-f_Ms[k])))
    p_Ms = np.array(p_Ms)
    p_Ms = p_Ms.reshape(-1, n_obj)


    if with_plots == 'y':
        plt.figure(figsize=(12,4))
        yll, mu, yul = np.percentile(p_Ms, [16,50,68], axis=0)
        plt.errorbar(np.arange(n_obj), mu, yerr=[mu-yll, yul-mu], fmt='.')
        plt.xlabel('Galaxy index')
        plt.ylabel(r'Probability of Being In A merger ($p_M$)')
        plt.title(r'Probability of a Galaxy Being in a Merger')


    return f_Ms, r_Is, r_Ms, classif, p_Ms
