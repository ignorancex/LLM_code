import numpy as np
import math
import matplotlib.pyplot as plt


class LearningRateDecay:
    def __init__(self, iteration, iteration_max=10000, lr=0.1, lr_min=1e-6):
        ''' learning rate decays with iteration. '''
        
        self.lr = lr
        self.lr_min = lr_min
        self.iteration = iteration
        self.iteration_max = iteration_max

    def exp(self, gamma=0.999, auto_params=True):
        '''
        exponential decay, return: lr * gamma^iteration
        
        :param auto_params: if True, gamma is set automatically        
        '''
        
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./self.iteration_max)
        lr_new = self.lr * gamma**self.iteration
        return lr_new
    
    def step(self, stepsize=1000, gamma=0.3, auto_params=True):
        '''
        learning rate decays step by step, similar to 'exp'
        
        :param auto_params: if True, gamma is set automatically        
        '''
        
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./(self.iteration_max*1.0/stepsize))
        lr_new = self.lr * gamma**(math.floor(self.iteration*1.0/stepsize))
        return lr_new
    
    def poly(self, decay_step=500, power=0.999, cycle=True):
        ''' polynomial decay, return: (lr-lr_min) * (1 - iteration/decay_steps)^power +lr_min '''
        
        if cycle:
            decay_steps = decay_step * math.ceil(self.iteration*1.0/decay_step)
        else:
            decay_steps = self.iteration_max
        lr_new = (self.lr-self.lr_min) * (1 - self.iteration*1.0/decay_steps)**power + self.lr_min
        return lr_new

#%% test
if __name__=='__main__':
    iteration = 10000
    lr_all_1 = []
    lr_all_2 = []
    lr_all_3 = []
    for ite in range(1, iteration+1):
        lr_all_1.append(LearningRateDecay(ite, iteration_max=iteration,lr=7.4e-5).exp(gamma=0.999, auto_params=True))
#        lr_all_2.append(LearningRateDecay(ite, iteration_max=iteration).step(stepsize=1000, gamma=0.3, auto_params=True))
#        lr_all_3.append(LearningRateDecay(ite, iteration_max=iteration,lr=0.01).poly(decay_step=500, power=1,cycle=True))
    
    lr_all_1 = np.array(lr_all_1)
#    lr_all_2 = np.array(lr_all_2)
#    lr_all_3 = np.array(lr_all_3)
    plt.figure(figsize=(8,6))
    plt.plot(lr_all_1, label='lr decay 1')
#    plt.plot(lr_all_2, label='lr decay 2')
#    plt.plot(lr_all_3, label='lr decay 3')
    plt.legend(fontsize=14)

