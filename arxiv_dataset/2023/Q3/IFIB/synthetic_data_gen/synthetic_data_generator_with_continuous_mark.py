# The synthetic data generator bases on Omi's FullyNN project.
# Derives from Omi's code in https://github.com/omitakahiro/NeuralNetworkPointProcess.

import numpy as np
import pandas as pd
import argparse, os
from functools import partial


from scipy.stats import lognorm
from scipy.special import erf


######################################################
### homogeneous possion process
######################################################
def generate_poisson(n):
    tau = np.random.exponential(scale = 4/3, size=n)
    T = tau.cumsum()
    intensity = np.ones_like(T)
    return T, tau, intensity


######################################################
### hawkes process
######################################################
def generate_hawkes1(n):
    [T,LL], intensity = simulate_hawkes(n,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL
    return T, score, intensity


def generate_hawkes2(n):
    [T,LL], intensity = simulate_hawkes(n,0.2,[0.4,0.4],[1.0,20.0])
    score = - LL
    return T, score, intensity


def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    Intensity = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            Intensity.append(l_next)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)], np.array(Intensity)


######################################################
### stationary renewal process
######################################################
def generate_stationary_renewal(n):
    s = np.sqrt(1)
    mu = -s*s/2
    tau = lognorm.rvs(s=s,scale=np.exp(mu),size=n)
    lpdf = -lognorm.logpdf(tau,s=s,scale=np.exp(mu))
    T = tau.cumsum()
    
    return T, lpdf

######################################################
### stationary renewal process
######################################################
def generate_stationary_renewal_intensity(n):
    s = np.sqrt(1)
    mu = 0
    tau = lognorm.rvs(s=s,scale=np.exp(mu),size=n)
    lpdf = -lognorm.logpdf(tau,s=s,scale=np.exp(mu))
    T = tau.cumsum()

    # Intensity function calculated by Wolfram Alpha
    Intensity = -0.797885*np.exp(-0.5*(np.log(tau))**2) / (-tau + tau * erf(0.707107 * np.log(tau)))
    
    return T, lpdf, Intensity

######################################################
### self-correcting process
######################################################
def generate_self_correcting(n):
    def self_correcting_process(mu,alpha,n):
        t, x = 0, 0
        T = []
        log_l = []
        Int_l = []
        Intensity = []
    
        for i in range(n):
            e = np.random.exponential()
            tau = np.log( e*mu/np.exp(x) + 1 )/mu # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t + tau
            T.append(t)
            x = x + mu*tau
            log_l.append(x)
            Intensity.append(np.exp(x))
            Int_l.append(e)
            x = x - alpha

        return [np.array(T),np.array(log_l),np.array(Int_l)], np.array(Intensity)
    
    [T,log_l,Int_l], intensity = self_correcting_process(1.5,1,n)
    score = -(log_l - Int_l)
    
    return T, score, intensity

def transform_autoregression(data_input, max_seq):
    data = np.array([[]])
    result = np.array([[]])
    score = np.array([[]])
    event = np.array([[]])
    intensity = np.array([[]])

    size = data_input.shape[0]
    for index in range(size):
        time = np.array(data_input.iloc[index].time_seq)
        L = np.array(data_input.iloc[index].score)
        event_seq = np.array(data_input.iloc[index].event)
        intensity_seq = np.array(data_input.iloc[index].intensity)
        try:
            assert (np.diff(time) < 0).any() == False
        except:
            raise ValueError("Non-monotonic increase time input detected.")

        time = np.diff(time, axis=-1)

        if time.shape[0] < max_seq + 1:
            time = np.concatenate(([0] * (max_seq - time.shape[0] + 1), time))

        for i in range(0, time.shape[0] - max_seq):
            if data.shape[1] == 0:
                data = time[i:i+max_seq].reshape(1, -1)
                result = time[i+max_seq].reshape(1, -1)
                score = L[i+max_seq].reshape(1, -1)
                event = event_seq[i+max_seq].reshape(1, -1)
                intensity = intensity_seq[i+max_seq].reshape(1, -1)
            else:
                data = np.append(
                    data, time[i:i+max_seq].reshape(1, -1), axis=0)
                result = np.append(
                    result, time[i+max_seq].reshape(1, -1), axis=0)
                score = np.append(
                    score, L[i+max_seq].reshape(1, -1), axis=0)
                event = np.append(
                    event, event_seq[i+max_seq].reshape(1, -1), axis=0)
                intensity = np.append(
                    intensity, intensity_seq[i+max_seq].reshape(1, -1), axis=0)


    return pd.DataFrame.from_dict({'data': data, 'result': result, 'score': score, 'event': event, 'intensity': intensity})


dataset_dict = {
    'hawkes_1': generate_hawkes1,
    'hawkes_2': generate_hawkes2,
    'self_correct': generate_self_correcting,
    'stationary_renewal': generate_stationary_renewal_intensity,
    'poisson': generate_poisson
}


def map_event_to_locations(num_classes, gen_seq_len, ndim):
    def event_gen_ex():    
        return np.random.randint(num_classes, size = gen_seq_len)

    # copied from neural_stpp by Chen et al.
    def pinwheel(num_samples, num_classes):
        radial_std = 0.3
        tangential_std = 0.1
        num_per_class = num_samples
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    
        features = np.random.randn(num_classes * num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)
    
        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
    
        return 2 * np.einsum("ti, tij -> tj", features, rotations)

    event = event_gen_ex()
    location_generation = partial(pinwheel, num_classes=num_classes)
    location_seq = location_generation(gen_seq_len)
    seq = np.zeros((gen_seq_len, ndim))
    for i, data_i in enumerate(np.split(location_seq, num_classes, axis=0)):
        seq = seq + data_i * np.expand_dims(i == event, axis = -1)
    
    return seq.tolist()


def data_gen(name, dataset, data_size, seq_len, num_classes, ndim, autoregression = False, data_with_event = False):

    data = {'time_seq': [], 'score': [], 'event': [], 'intensity': []}
    if autoregression:
        gen_seq_len = seq_len * 2
    else:
        gen_seq_len = seq_len
    for _ in range(data_size):
        '''
        Event distribution is time-agnostic.
        '''
        # data['index'].append(i)
        time, score, intensity = dataset_dict[dataset](gen_seq_len)
        data['time_seq'].append(time.tolist())
        data['score'].append(score.tolist())
        data['intensity'].append(intensity.tolist())
        data['event'].append(map_event_to_locations(num_classes, gen_seq_len, ndim))

    final = pd.DataFrame.from_dict(data)
    if autoregression:
        final = transform_autoregression(final, seq_len)
    final.to_json(os.path.join('.', f'{opt.dataset_name}_continuous', name + ('_auto' if autoregression else '') + '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=['hawkes_1', 'hawkes_2', 'poisson', 'self_correct', \
                                                             'stationary_renewal', 'hawkes_and_self_correcting', 'hawkes_and_poisson'],
                        help="How to generate synthetic temporal point process data.")
    parser.add_argument('--data_type', type=str, choices=['autoregression', 'sequence'],
                        help='Autoregression: Each line of data comprise three parts: history: relative time sequence of history events,\
                              result: the time interval between this event and the last event in history, score: the ideal value of log(p).\n \
                              Sequence: Each line of data comprise two parts: event time sequence, mask sequence and score sequence. Ceveats: the timestamps in sequence\
                              datasets are absolute.')
    parser.add_argument('--seq_length', type=int, help="For autoregression datasets, it is the length of history sequence. \
                                                        For sequence datasets: it is the length of sequence")
    parser.add_argument('--train_size', type=int, help='The size of training dataset')
    parser.add_argument('--eva_size', type=int, help='The size of evaluation dataset')
    parser.add_argument('--test_size', type=int, help='The size of test dataset')
    parser.add_argument('--num_classes', type=int, help='The number of available classes.')
    parser.add_argument('--ndim', type = int, help='How many dimension the continuous marker space would have.')
    parser.add_argument('--random_seed', type=int, help='Global random seed.')

    opt = parser.parse_args()

    if not os.path.exists(os.path.join('.', f'{opt.dataset_name}_continuous')):
        os.mkdir(os.path.join('.', f'{opt.dataset_name}_continuous'))
    
    with open(f'./{opt.dataset_name}_continuous/num_events.txt', 'w') as f:
        f.write(str(opt.ndim))

    np.random.seed(opt.random_seed)

    if opt.data_type == 'autoregression':
        data_gen('train', opt.dataset_name, opt.train_size, opt.seq_length, opt.num_classes, opt.ndim, True)
        data_gen('evaluate', opt.dataset_name, opt.eva_size, opt.seq_length, opt.num_classes, opt.ndim, True)
        data_gen('test', opt.dataset_name, opt.test_size, opt.seq_length, opt.num_classes, opt.ndim, True)
    else:
        data_gen('train', opt.dataset_name, opt.train_size, opt.seq_length, opt.num_classes, opt.ndim)
        data_gen('evaluate', opt.dataset_name, opt.eva_size, opt.seq_length, opt.num_classes, opt.ndim)
        data_gen('test', opt.dataset_name, opt.test_size, opt.seq_length, opt.num_classes, opt.ndim)