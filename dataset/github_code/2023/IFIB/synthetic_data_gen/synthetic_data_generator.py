# The synthetic data generator bases on Omi's FullyNN project.
# Derives from Omi's code in https://github.com/omitakahiro/NeuralNetworkPointProcess.

import numpy as np
import pandas as pd
import argparse, os

from scipy.stats import lognorm
from scipy.special import erf

######################################################
### homogeneous possion process
######################################################
def generate_poisson(n):
    tau = np.random.exponential(scale = 1, size = n)
    T = tau.cumsum()
    intensity = np.ones_like(T)
    return T, tau, intensity

######################################################
### hawkes process
######################################################
def generate_hawkes1(n):
    [T,LL], intensity = simulate_hawkes(n, 0.2, [0.8, 0.0], [1.0, 20.0])
    score = - LL
    return T, score, intensity

def generate_hawkes2(n):
    [T,LL], intensity = simulate_hawkes(n, 0.2, [0.4, 0.4], [1.0, 20.0])
    score = - LL
    return T, score, intensity

def simulate_hawkes(n, mu, alpha, beta):
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
        
        l_trg_Int1 += l_trg1*(1 - np.exp(-beta[0]*step)) / beta[0]
        l_trg_Int2 += l_trg2*(1 - np.exp(-beta[1]*step)) / beta[1]
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
    tau = lognorm.rvs(s = s,scale = np.exp(mu), size = n)
    lpdf = -lognorm.logpdf(tau,s = s, scale = np.exp(mu))
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
    
    [T,log_l,Int_l], intensity = self_correcting_process(1, 1, n)
    score = -(log_l - Int_l)
    
    return T, score, intensity

######################################################
### hawkes process + self-correcting process
######################################################
def generate_hawkes_and_self_correcting(n):
    '''
    2022-03-04
    Now, the true intensity and score for all mixed temporal point processes is not available.
    '''
    T_self_correcting, score_self_correcting, intensity_self_correcting = generate_self_correcting(n//2)
    T_hawkes, score_hawkes, intensity_hawkes = generate_hawkes1(n//2)
    event = np.concatenate((np.zeros(n//2), np.ones(n//2)), axis = -1)

    # sort the array
    T_original = np.concatenate((T_self_correcting, T_hawkes), axis = -1)
    index = np.argsort(T_original)

    T = T_original[index]
    event = event[index]
    score = np.zeros_like(T)
    intensity = np.zeros_like(T)

    return T, score, intensity, event

######################################################
### hawkes process + poisson process
######################################################
def generate_hawkes_and_poisson(n):
    '''
    2022-03-04
    Now, the true intensity and score for all mixed temporal point processes is not available.
    '''
    T_self_correcting, score_self_correcting, intensity_self_correcting = generate_self_correcting(n//2)
    T_hawkes, score_hawkes, intensity_hawkes = generate_poisson(n//2)
    event = np.concatenate((np.zeros(n//2), np.ones(n//2)), axis = -1)

    # sort the array
    T_original = np.concatenate((T_self_correcting, T_hawkes), axis = -1)
    index = np.argsort(T_original)

    T = T_original[index]
    event = event[index]
    score = np.zeros_like(T)
    intensity = np.zeros_like(T)

    return T, score, intensity, event

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
    'poisson': generate_poisson,
    'hawkes_and_self_correcting': generate_hawkes_and_self_correcting,
    'hawkes_and_poisson': generate_hawkes_and_poisson
}

def data_gen(name, dataset, data_size, seq_len, autoregression = False, data_with_event = False):
    data = {'time_seq': [], 'score': [], 'event': [], 'intensity': []}
    if autoregression:
        gen_seq_len = seq_len * 2
    else:
        gen_seq_len = seq_len
    for i in range(data_size):
        if data_with_event:
            '''
            Event distribution is time-aware.
            '''
            time, score, intensity, event = dataset_dict[dataset](gen_seq_len)
            data['time_seq'].append(time.tolist())
            data['score'].append(score.tolist())
            data['intensity'].append(intensity.tolist())
            data['event'].append(event.tolist())
        else:
            '''
            Event distribution is time-agnostic.
            '''
            # data['index'].append(i)
            time, score, intensity = dataset_dict[dataset](gen_seq_len)
            data['time_seq'].append(time.tolist())
            data['score'].append(score.tolist())
            data['intensity'].append(intensity.tolist())
            data['event'].append(event_gen(size = gen_seq_len, time = time))
    
    final = pd.DataFrame.from_dict(data)
    if autoregression:
        final = transform_autoregression(final, seq_len)
    final.to_json(os.path.join('.', opt.dataset_name, name + ('_auto' if autoregression else '') + '.json'))

def event_gen_ex(size, time):    
    return np.random.randint(5, size = size)

def event_gen(size, time):
    time_diff = np.diff(np.concatenate([np.array([0]), time]))
    time_diff_mean = time_diff.mean()
    time_diff_std = time_diff.std()

    '''
    the length of intervals smaller than time_average - time_standard_variance: type 0
    [time_average - time_standard_variance, time_average - 0.5 * time_standard_variance]: type 1
    [time_average - 0.5 time_standard_variance, time_average]: type 2
    [time_average, time_average + 0.5 * time_standard_variance]: type 3
    [time_average + 0.5 * time_standard_variance, time_average + time_standard_variance]: type 4
    the length of intervals larger than time_average + time_variance: type 5
    '''
    events = np.zeros(size)
    events[time_diff < time_diff_mean - time_diff_std] = 0
    events[((time_diff_mean - time_diff_std) <= time_diff) & (time_diff < (time_diff_mean - 0.5 * time_diff_std))] = 1
    events[((time_diff_mean - 0.5 * time_diff_std) <= time_diff) & (time_diff < time_diff_mean)] = 2
    events[(time_diff_mean < time_diff) & (time_diff <= (time_diff_mean + 0.5 * time_diff_std))] = 3
    events[((time_diff_mean + 0.5 * time_diff_std) <= time_diff) & (time_diff < (time_diff_mean + time_diff_std))] = 4
    events[time_diff_mean + time_diff_std <= time_diff] = 5

    return events

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

    parser.add_argument('--random_seed', type=int, help='Global random seed.')

    opt = parser.parse_args()

    data_with_event = ['hawkes_and_self_correcting', 'hawkes_and_poisson']
    data_with_event_mark = False
    if opt.dataset_name in data_with_event:
        data_with_event_mark = True

    if not os.path.exists(os.path.join('.', opt.dataset_name)):
        os.mkdir(os.path.join('.', opt.dataset_name))
    
    with open(f'./{opt.dataset_name}/num_events.txt', 'w') as f:
        f.write(str(2) if data_with_event else str(6))

    np.random.seed(opt.random_seed)

    if opt.data_type == 'autoregression':
        data_gen('train', opt.dataset_name, opt.train_size, opt.seq_length, True, data_with_event = data_with_event_mark)
        data_gen('evaluate', opt.dataset_name, opt.eva_size, opt.seq_length, True, data_with_event = data_with_event_mark)
        data_gen('test', opt.dataset_name, opt.test_size, opt.seq_length, True, data_with_event = data_with_event_mark)
    else:
        data_gen('train', opt.dataset_name, opt.train_size, opt.seq_length, data_with_event = data_with_event_mark)
        data_gen('evaluate', opt.dataset_name, opt.eva_size, opt.seq_length, data_with_event = data_with_event_mark)
        data_gen('test', opt.dataset_name, opt.test_size, opt.seq_length, data_with_event = data_with_event_mark)