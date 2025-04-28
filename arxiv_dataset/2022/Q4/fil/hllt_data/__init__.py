# Taken from KIV code release / deep/data_generator.py, which in turn is from the DeepIV repo

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as onp 
from .mnist import load_mnist_realval
#from causenet.datastream import DataStream, prepare_datastream
#from sklearn.preprocessing import OneHotEncoder

X_mnist = None
y_mnist = None


def loadmnist():
    '''
    Load the mnist data once into global variables X_mnist and y_mnist.
    '''
    global X_mnist
    global y_mnist
    train, val, test = load_mnist_realval(
        os.path.expanduser('~/run/mnist.pkl.gz'), one_hot=False)
    test = (onp.concatenate([test[0], val[0]], 0), onp.concatenate([test[1], val[1]], 0))
    X_mnist = []
    y_mnist = []
    for d in [train, test]:
        X, y = d
        idx = onp.argsort(y)
        X_mnist.append(X[idx, :])
        y_mnist.append(y[idx])

def get_images(digit, n, seed=None, testset=False):
    """
    return a float tensor of shape [n, 784]
    """
    if X_mnist is None:
        loadmnist()
    is_test = int(testset)
    rng = onp.random.RandomState(seed)
    X_i = X_mnist[is_test][y_mnist[is_test] == digit]
    n_i, shp = X_i.shape
    assert shp == 28 * 28
    perm = rng.permutation(onp.arange(n_i))
    if n > n_i:
        # raise ValueError('You requested %d images of digit %d when there are \
	# 					  only %d unique images in the %s set.' % (n, digit, n_i, 'test' if testset else 'training'))
        p0 = perm
        while perm.shape[0] < n:
            perm = onp.concatenate([perm, p0])
    return X_i[perm[:n]]

def one_hot(col, n_values):
    z = col.reshape(-1,1)
    ret = onp.zeros((z.shape[0], n_values))
    for i in range(n_values):
        ret[z[:,0] == i, i] = 1
    return ret

def sensf(x):  # equiv to aux/get_psi.m
    return 2.0*((x - 5)**4 / 600 + onp.exp(-((x - 5)/0.5)**2) + x/10. - 2)

def emocoef(emo):
    emoc = (emo * onp.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
    return emoc

psd = 3.7
pmu = 17.779
ysd = 158.#292.
ymu = -292.1

def storeg(x, price):
    """
    f=@(p,t,s) 100+(10+p).*s.*get_psi(t)-2.*p; % Hartford Lewis Leyton-Brown Taddy
    """
    emoc = emocoef(x[:, 1:])
    time = x[:, 0]
    g = sensf(time)*emoc*10. + (emoc*sensf(time)-2.0)*(psd*price.flatten() + pmu)
    y = (g - ymu)/ysd
    return y.reshape(-1, 1)

def encode_image(emotion_id, seed, test):
    idx = onp.argsort(emotion_id)
    emotion_feature = onp.zeros((0, 28*28))
    for i in range(7):
        img = get_images(i, onp.sum(emotion_id == i), seed, test)
        emotion_feature = onp.vstack([emotion_feature, img])
    reorder = onp.argsort(idx)
    return emotion_feature[reorder, :]

def demand(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, use_images=False, test=False, use_hllt=False):
    global psd, pmu, ysd, ymu    
    if use_hllt:
        psd = ysd = 1
        pmu = ymu = 0
    else:
        psd = 3.7
        pmu = 17.779
        ysd = 158.#292.
        ymu = -292.1
    # KernelIV paper uses ynoise=pnoise=1. 
    rng = onp.random.RandomState(seed)

    # covariates: time and emotion
    time = rng.rand(n) * 10
    emotion_id = rng.randint(0, 7, size=n)
    emotion = one_hot(emotion_id, n_values=7)
    if use_images:
        emotion_feature = encode_image(emotion_id, seed, test)
    else:
        emotion_feature = emotion

    # random instrument
    z = rng.randn(n)

    # z -> price.  p=25+(z+3).*get_psi(t)+v
    v = rng.randn(n)*pnoise
    price = sensf(time)*(z + 3)  + 25.
    price = price + v
    price = (price - pmu)/psd

    # true observable demand function
    x = onp.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
    x_latent = onp.concatenate([time.reshape((-1, 1)), emotion], axis=1)
    g = lambda x, z, p: storeg(x, p) # doesn't use z

    # errors 
    e = (ypcor*ynoise/pnoise)*v + rng.randn(n)*ynoise*onp.sqrt(1-ypcor**2)
    e = e.reshape(-1, 1)
    
    # response
    y = g(x_latent, None, price) + e

    return (x,
            z.reshape((-1, 1)),
            price.reshape((-1, 1)),
            y.reshape((-1, 1)),
            g)


def linear_data(n, seed=None, sig_d=0.5, sig_y=2, sig_t=1.5,
				alpha=4, noiseless_t=False, **kwargs):
    rng = onp.random.RandomState(seed)
    nox = lambda z, d: z + 2*d
    house_price = lambda alpha, d, nox_val: alpha + 4*d + 2*nox_val

    d = rng.randn(n) * sig_d
    law = rng.randint(0, 2, n)

    if noiseless_t:
        t = nox(law, d.mean()) + sig_t*rng.randn(n)
    else:
        t = (nox(law, d) + sig_t*rng.randn(n) - 0.5) / 1.8
    z = law.reshape((-1, 1))
    x = onp.zeros((n, 0))
    y = (house_price(alpha, d, t) + sig_y*rng.randn(n) - 5.)/5.
    g_true = lambda x, z, t: house_price(alpha, 0, t)
    return x, z, t.reshape((-1, 1)), y.reshape((-1, 1)), g_true


def gen_hllt_data(N, cor, seed, use_im=False):
    """
    return: (Dtrain, Dtest), true_fn, ysd
    NOTE: when use_im is True, true_fn will take a 3-dim (latent) input, or a 786-dim input
          *only if* it is exactly Xtest. This is sufficient for our NN code
    z = [supply]; w = [emo; time]; x = [price]
    """
    Ztr, Wtr, Xtr, Ytr, true_fn, ysd = _hllt_gen_data(N, cor, seed, use_im)

    # gen test
    eax = onp.array(list(range(7))) + 1
    tax = onp.linspace(0, 10, 20)
    pax = onp.linspace(10, 25, 20)
    Xtest = [a.reshape((-1, 1)) for a in onp.meshgrid(eax, tax, pax)]
    Xtest_latent = Xtest.copy()
    if use_im:
        Xtest[0] = encode_image((Xtest[0]-1).squeeze(1), seed, test=True) 
    Wtest = onp.concatenate(Xtest[:2], -1)
    Xtest = Xtest[-1]
    if use_im:  # hacks true_fn
        f0 = true_fn
        def true_fn_wrapped(W, X):
            if W.shape[-1] == 2:
                return f0(W, X)
            else:
                assert W.shape == Wtest.shape and onp.all(onp.abs(Wtest-W) < 1e-5)
                return f0(onp.concatenate(Xtest_latent[:2], -1), Xtest_latent[2])
        true_fn = true_fn_wrapped

    return (Ztr, Wtr, Xtr, Ytr), (None, Wtest, Xtest, None), true_fn, ysd


def _hllt_gen_data(ssz, cor, seed, use_im):
    """
    This corresponds to the "HLLT design" in the KIV (and DualIV) experiments, which removes
    the normalization on both Y and price.
    """
    W, Z, Price, Y, g = demand(ssz, seed, ypcor=cor, use_hllt=True, use_images=use_im)
    time, emo = W[:, :1], W[:, 1:]
    if not use_im:  # following KIV, additionally change the one hot encoding
        emo = emocoef(emo)[:, None]
    W = onp.concatenate([emo, time], axis=-1)
    ymean, ysd = Y.mean(), Y.std()  # observable thus valid
    # ysd /= 10
    def true_fn(w, p):
        assert w.shape[1] == 2
        time, emo = w[:, 1:], (w[:, :1] - 1).astype('i')
        w = onp.concatenate([time, one_hot(emo, 7)], axis=-1)
        return (g(w, None, p) - ymean) / ysd
    return Z, W, Price, (Y-ymean)/ysd, true_fn, ysd

