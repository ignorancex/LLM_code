import numpy as np
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import constant, RandomUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import warnings

### PIM for regression problems (including time series)
class PIM(Layer):

    def __init__(self, p, err_type, beta, **kwargs):
        super(PIM, self).__init__(**kwargs)
        self.p = p
        self.err_type = err_type
        self.beta = beta

    def build(self, input_shape):
        # Is the error distribution symmetric or asymmetric?
        if self.err_type == 'sym':
            self.picp = self.picp_sym
            shape = (input_shape[1],)
        elif self.err_type == 'asym':
            self.picp = self.picp_asym
            shape = (input_shape[1], 2)
        # Do we add the conformal correction to p?
        if self.conformal: 
            self.p *= (1 + 1/input_shape[0]) # batch_size = m
        
        # Initialize prediction interval normal z-scores
        rad0 = norm.ppf(0.5*(1 + self.p)) /np.sqrt(5000)
        self.rad = self.add_weight(shape=shape, name = 'rad',
                                    initializer=constant(value=rad0),
                                    dtype = K.floatx(),
                                    trainable=True)
        self.built = True

    def picp_sym(self, err):
        return K.mean(K.sigmoid(self.beta*(self.rad - K.abs(err))), axis=0)

    def picp_asym(self, err):
        err_l = tf.where(err <= 0, err, self.rad[:,0] * K.ones_like(err))
        err_u = tf.where(err > 0, err, self.rad[:,1] * K.ones_like(err))
        is_lower = K.sigmoid(self.beta*(self.rad[:,0] - K.abs(err_l)))
        is_upper = K.sigmoid(self.beta*(self.rad[:,1] - K.abs(err_u)))
        is_lower = tf.where(K.abs(is_lower-0.5)>1e-8, is_lower, K.zeros_like(err))
        is_upper = tf.where(K.abs(is_upper-0.5)>1e-8, is_upper, K.zeros_like(err))
        picp_l = K.sum(is_lower,0) / (K.cast(tf.math.count_nonzero(is_lower,0), K.floatx())+1e-8)
        picp_u = K.sum(is_upper,0) / (K.cast(tf.math.count_nonzero(is_upper,0), K.floatx())+1e-8)
        return K.concatenate((picp_l, picp_u))

    def call(self, inputs):
        return self.picp(inputs)

    @property
    def loss(self):
        ''' ypred is the picp computed by PIM '''
        return lambda _, picp: K.square(picp - self.p)

    @property
    def radius(self):
        ''' Radii of lower and upper ends of PI'''
        rad = K.eval(self.rad)
        if self.err_type == 'sym':
            return np.squeeze(rad), np.squeeze(rad)
        else:
            return np.squeeze(rad[:,0]), np.squeeze(rad[:,1])

    def get_config(self):
        config = super(PIM, self).get_config()
        config.update({'p': self.p})

class PredictionInterval(Model):

    def __init__(self, p, err_type='sym', beta=1e3, **kwargs):
        super(PredictionInterval, self).__init__()
        self.pim = PIM(p, err_type, beta, **kwargs)
        self.early_stop = EarlyStopping(monitor='loss', 
                                        mode='min', 
                                        patience=5)
    def call(self, inputs):
        return self.pim(inputs)

### PIM for classification problems
class PIMc(Layer):

    def __init__(self, p, num_classes, beta, threshold, **kwargs):
        self.p = p
        self.beta = beta
        self.nc = num_classes if num_classes !=2 else num_classes-1
        # Confidence interval for the mean of U(a,b) with b-a = 1/2
        self.mpiw_unif = 0.5 * self.p
        self.mpiw0 = np.random.uniform(1e-8, self.mpiw_unif,(4,self.nc))
        if threshold == 'fix':
            self.thresh = 0.5
            self.picp = self.picp_fix_thresh
        else:
            thresholds = np.arange(0.1,0.9,0.1)
            self.thresh = K.expand_dims(K.cast(thresholds, K.floatx()),0) 
            self.picp = self.picp_var_thresh
        super(PIMc, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mpiw = self.add_weight(shape=(4, self.nc), name = 'ci_mpiw', 
                                    initializer=constant(value=self.mpiw0),
                                    dtype = K.floatx(),
                                    trainable=True)
        self.mpiw_not_found = np.array(4*[True])
        self.built = True
    
    def picp_fix_thresh(self, ytrue, ypred):
        dist2true = K.abs(ytrue - ypred)
        ones, zeros = K.ones_like(ytrue), K.zeros_like(ytrue)
        # False or true predictions
        ft = tf.where(dist2true < 0.5, ones, zeros)
        # Map to indices [0,1,2,3] corresponding to [FN,FP,TN,TP]: bin->dec
        S = tf.one_hot(K.cast(2*ft + ytrue, 'int32'), 4, axis=1, dtype=K.floatx())
        # S is a ~ one-hot-encoded tensor of shape (batch, 2*num_classes)
        # Use it to select the active weights per sample
        active_mpiw = K.sum(K.expand_dims(self.mpiw, 0) * S, 1) 
        # False negative or false positive
        dist_ftp = ft * dist2true + (1-ft) * K.abs(0.5 - ypred)
        in_or_out = K.sigmoid(self.beta*(active_mpiw - dist_ftp)) 
        # Keep count of in_or_out for each of FN, FP, TN, TP
        in_or_out = K.tile(K.expand_dims(in_or_out, 1), (1,4,1)) * S
        # Compute probability. If self.mpiw are optimal, we must have p ~ self.p
        return K.sum(in_or_out, 0) / (K.sum(S, 0) + K.epsilon())

    def picp_var_thresh(self, ytrue, ypred):
        ytrue = K.expand_dims(K.squeeze(ytrue,-1), 1)
        ypred = K.expand_dims(K.squeeze(ypred,-1), 1)
        # False or true predictions
        tp = (ypred > self.thresh) & (K.cast(ytrue, 'int32') == 1)
        tn = (ypred < self.thresh) & (K.cast(ytrue, 'int32') == 0)
        ft = tf.where(tp | tn, K.ones_like(ytrue), K.zeros_like(ytrue))
        # Map to indices [0,1,2,3] corresponding to [FN,FP,TN,TP]: bin->dec
        S = tf.one_hot(K.cast(2*ft + ytrue, 'int32'), 4, axis=2, dtype=K.floatx())
        # S is a ~ one-hot-encoded tensor of shape (batch, 2*num_classes)
        # Use it to select the active weights per sample
        active_mpiw = K.sum(K.reshape(self.mpiw, (1,1,4)) * S, -1) 
        # Error distance from preds to targets
        dist2targ = ft * K.abs(ytrue - ypred) + (1-ft) * K.abs(self.thresh - ypred)
        in_or_out = K.sigmoid(self.beta*(active_mpiw - dist2targ)) 
        # Keep count of in_or_out for each of FN, FP, TN, TP
        in_or_out = K.tile(K.expand_dims(in_or_out, -1), (1,1,4)) * S
        # Compute probability. If self.mpiw are optimal, we must have p ~ self.p
        return K.sum(in_or_out, (0,1)) / (K.sum(S, (0,1)) + K.epsilon())

    def call(self, inputs):
        # Count ytrues in confidence interval 
        ytrue, ypred = inputs
        return self.picp(ytrue, ypred)

    @property
    def loss(self):
        ''' ypred is the picp computed by PIM '''
        return lambda ytrue, ypred: K.square(ypred - self.p)
    
    @property
    def urates(self):
        # piw of rates: FN, FP, TN, TP
        mpiw = K.eval(self.mpiw)
        mpiw0 = K.eval(self.mpiw0)
        eps = K.eval(K.epsilon())
        self.mpiw_not_found = np.abs(mpiw - mpiw0) <= eps
        if self.mpiw_not_found.any(): 
            msg = f'PIM did not evolve in some cases!'
            warnings.warn(msg)
            mpiw[self.mpiw_not_found] = None
        # PIM learns mpiw, uncertainty is half of that
        return 0.5 * mpiw

    def binary_umetrics(self, ypred, ytrue):
        # After the model is fit
        # Uncertainty to priors
        ytrue, ypred = np.squeeze(ytrue), np.squeeze(np.round(ypred))
        inds_pos, inds_neg = (ytrue == 1), (ytrue == 0) 
        sum_pos, sum_neg = np.sum(inds_pos), np.sum(inds_neg)
        fnr = np.sum(ypred[inds_pos] == 0) / sum_pos
        fpr = np.sum(ypred[inds_neg] == 1) / sum_neg
        tnr = 1-fpr
        tpr = 1-fnr
        ufnr, ufpr, utnr, utpr = np.squeeze(self.urates)
        # Uncertainty to posteriors
        inds_ppos, inds_pneg = (ypred == 1), (ypred == 0)
        npv = np.sum(ytrue[inds_pneg] == 0) / np.sum(inds_pneg)
        ppv = np.sum(ytrue[inds_ppos] == 1) / np.sum(inds_ppos)
        p_1 = sum_pos / ytrue.shape[0]
        p_0 = 1 - p_1
        imb = p_0 / p_1
        iimb = 1/imb
        uppv = imb * (ppv**2) * ((fpr/tpr)*(utpr/tpr) + ufpr/tpr + imb*ppv*((ufpr/tpr)**2))
        unpv = iimb * (npv**2) * ((fnr/tnr)*(utnr/tnr) + ufnr/tnr + iimb*npv*((ufnr/tnr)**2))
        # Accuracy and uncertainty
        acc = tnr * p_0 + tpr * p_1
        uacc = utnr * p_0 + utpr * p_1
        # F1 score and uncertainty
        f1 = 2*(ppv * tpr)/(ppv + tpr)
        uf1_dtpr = (f1*(1+imb*(fpr/tpr)*ppv)+0.5*(f1**2/ppv)*(1+imb*fpr*(ppv/tpr)**2))*(utpr/tpr)
        uf1_dfpr = imb*f1*ppv*(0.5*f1/tpr-1)*(ufpr/tpr)
        uf1_d2fpr = 0.5*(imb**2)*f1*(ppv**2)*((0.5*f1-tpr)**2 +\
                    0.5*((f1/tpr)**2*(0.5-tpr/f1)-f1/tpr)+1)*(ufpr/tpr)**2
        uf1 = abs(uf1_dtpr) + abs(uf1_dfpr) + abs(uf1_d2fpr)        
        res = {'TPR': [tpr, utpr],
               'FPR': [fpr, ufpr],
               'PPV': [ppv, uppv],
               'NPV': [npv, unpv],
               'F1': [f1, uf1],
               'ACC': [acc, uacc],
               }
        return res
    
    def categorical_acc(self, ypred, ytrue):
        ypred = np.round(ypred)
        def tpr_per_class(k):
            inds_pos = (ytrue[:,k] == 1)
            tpr_k = np.sum(ypred[inds_pos,k] == 1,0) / np.sum(inds_pos,0)
            return tpr_k
        tpr = np.array([tpr_per_class(k) for k in range(ytrue.shape[1])])
        # Accuracy and uncertainty
        p_1 = np.mean(ytrue == 1,0)
        acc = np.sum(tpr * p_1)
        _, _, _, utpr = self.urates
        uacc = np.sum(utpr * p_1)
        return acc, uacc

class ConfidenceInterval(Model):

    def __init__(self, p, num_classes, beta=1e3, threshold='fix', **kwargs):
        super(ConfidenceInterval, self).__init__()
        self.pim = PIMc(p, num_classes, beta, threshold, **kwargs)
        self.early_stop = EarlyStopping(monitor='loss', 
                                        mode='min', 
                                        patience=5)
    def call(self, inputs):
        return self.pim(inputs)

if __name__ == '__main__':
    ### Regression
    # y_true = np.random.normal(loc=0., scale=1.0, size=(1000,1))
    # y_pred = np.zeros_like(y_true)
    # mu = 0
    a, b, mu = 1, 0.5, 3./4
    y_true = np.random.beta(a, b, (5000,1))
    y_pred = mu * np.ones_like(y_true)

    err = y_true - y_pred
    umodel = PredictionInterval(0.95, err_type='asym')
    umodel.compile(loss=umodel.pim.loss, optimizer=Adam(lr=0.005))
    hist = umodel.fit(err, err,
                    epochs=4000,
                    batch_size=100,
                    callbacks=[umodel.early_stop],
                    verbose=0)
    plt.plot(hist.history['loss'])
    plt.show()
    rad_l, rad_u = umodel.pim.radius
    print(f'PI estimated:{(mu-rad_l, mu+rad_u)}')
    # print(f'PI theoretical: {norm.interval(0.95)}')
    print(f'PI theoretical: {beta.interval(0.95, a, b)}')
    print(f'PICP = {umodel.pim.picp(err.astype(np.float32))}')
