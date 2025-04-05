#!/usr/bin/env python
# coding: utf-8

from utils import *

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Parameter, optimizers, cuda
import cupy as cp


class Lnr(chainer.Chain):
    def __init__(self, dim, prior, loss='sq'):
        super(Lnr, self).__init__()
        with self.init_scope():
            self.d = dim
            self.p = prior
            self.n = 1 - prior
            self.ls = loss
            if dim < 3000:
                self.l1 = L.Linear(None, 100)
                self.l2 = L.Linear(None, 1)
            else:
                self.conv1 = L.Convolution2D(3, 32, 3, pad=1)
                self.conv2 = L.Convolution2D(32, 32, 3, pad=1)
                self.conv3 = L.Convolution2D(32, 32, 3, pad=1)
                self.conv4 = L.Convolution2D(32, 32, 3, pad=1)
                self.conv5 = L.Convolution2D(32, 32, 3, pad=1)
                self.conv6 = L.Convolution2D(32, 32, 3, pad=1)
                self.l1 = L.Linear(512, 512)
                self.l2 = L.Linear(512, 1)

    def f(self, x):
        if self.d < 3000:
            return self.l2(F.relu(self.l1(x)))
        else:
            h = F.relu(self.conv1(x))
            h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
            h = F.relu(self.conv3(h))
            h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
            h = F.relu(self.conv5(h))
            h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
            h = F.relu(self.l1(h))
            return self.l2(h)

    def prepare(self, Xt1, Xt2):
        X1a = Xt1[:, :self.d]
        X1b = Xt1[:, self.d:(2*self.d)]
        X1c = Xt1[:, (2*self.d):]
        X2a = Xt2[:, :self.d]
        X2b = Xt2[:, self.d:(2*self.d)]
        X2c = Xt2[:, (2*self.d):]
        Xa = self.xp.concatenate((X1a, X1c, X2a, X2c))
        Xb = X1b
        Xc = X2b
        if self.d > 3000:
            Xa = Xa.reshape((-1, 3, 32, 32))
            Xb = Xb.reshape((-1, 3, 32, 32))
            Xc = Xc.reshape((-1, 3, 32, 32))
        p, n = self.p, self.n
        deno = 1 - p * n
        A = (p*p*p + 2*p*p*n) / deno
        B = (2*p*n*n + n*n*n) / deno
        return Xa, Xb, Xc, A, B

    def loss(self, z):
        if self.ls == 'sq':
            return ((z - 1) ** 2) / 4
        elif self.ls == 'dh':
            zeros = self.xp.zeros(z.shape, dtype=self.xp.float32)
            return F.maximum(-z, F.maximum(zeros, (1-z)/2))
        elif self.ls == 'lg':
            return F.log(1 + F.exp(-z)) / self.xp.log(2)
        elif self.ls == 'exp':
            return F.exp(-z)

    def risk(self, Xt1, Xt2):
        Xa, Xb, Xc, A, B = self.prepare(Xt1, Xt2)
        p, n = self.p, self.n
        a = A*A + p*p + n*n
        b = A*B + 2*p*n
        c = B*B + p*p + n*n
        coe = 1 / (a*c - b*b)
        r_a = p*(c*p-b*n) * self.loss(self.f(Xa)) + \
              n*(a*n-b*p) * self.loss(-self.f(Xa))
        r_b = p*(c*A-b*B) * self.loss(self.f(Xb)) + \
              n*(a*B-b*A) * self.loss(-self.f(Xb))
        r_c = p*(c*n-b*p) * self.loss(self.f(Xc)) + \
              n*(a*p-b*n) * self.loss(-self.f(Xc))
        return coe * (F.average(r_a) + F.average(r_b) + F.average(r_c))

    def test(self, Xs, ys):
        res = sum((self.f(Xs).data * ys) >= 0)[0] / len(Xs)
        return res


def subrun(X1train, X1valid, X2train, X2valid, prior, decay_rate, dim, args):
    model = Lnr(dim=dim, prior=prior, loss=args.l)
    model.to_gpu()
    opt = optimizers.Adam(weight_decay_rate=decay_rate)
    opt.setup(model)
    for epoch in range(args.epochs):
        model.cleargrads()
        risk = model.risk(X1train, X2train)
        risk.backward()
        opt.update()
    return model.risk(X1valid, X2valid).data


def run(dataset, l, dim, nt, args):
    decay_rates = [1e-6, 1e-4, 1e-2]
    Xt1, Xt2, Xtest, ytest = gen_dataset(nt=nt,
                                         prior=args.p,
                                         dataset=args.dataset)
    n1, n2 = len(Xt1), len(Xt2)
    prior = (1 + np.sqrt(1 - 4 * (1 - n1 / (n1 + n2)))) / 2

    min_rsk = 1e30
    for decay_rate in decay_rates:
        rsk = 0
        for train1, valid1 in KFold(n_splits=3).split(Xt1):
            for train2, valid2 in KFold(n_splits=3).split(Xt2):
                Xt1_train = cp.asarray(Xt1[train1])
                Xt1_valid = cp.asarray(Xt1[valid1])
                Xt2_train = cp.asarray(Xt2[train2])
                Xt2_valid = cp.asarray(Xt2[valid2])
                rsk += subrun(Xt1_train, Xt1_valid, Xt2_train, Xt2_valid,
                              prior, decay_rate, dim, args)
        printr('{} {}'.format(decay_rate, rsk))
        if rsk <  min_rsk:
            min_rsk, min_rate = rsk, decay_rate
    printr('{:.3f} {}'.format(prior, min_rate))
    model = Lnr(dim=dim, prior=prior, loss=l)
    model.to_gpu()
    optimizer = chainer.optimizers.Adam(weight_decay_rate=min_rate)
    optimizer.setup(model)
    Xt1 = model.xp.asarray(Xt1)
    Xt2 = model.xp.asarray(Xt2)
    Xtest, ytest = model.xp.asarray(Xtest), model.xp.asarray(ytest)
    for epoch in range(args.epochs):
        model.cleargrads()
        risk = model.risk(Xt1, Xt2)
        risk.backward()
        optimizer.update()
    test = model.test(Xtest, ytest)
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('-p', type=float, default=0.7)
    parser.add_argument('--dataset', '-d', default='toy')
    parser.add_argument('--runs', '-r', type=int, default=20)
    parser.add_argument('--nt', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-l', type=str, default='dh') # 'sq', 'lg', 'exp'
    args = parser.parse_args()

    if args.dataset == 'toy':
        dim = 2
    cuda.get_device(args.gpu).use()

    print('{} {} {} {}'.format(args.dataset, args.l, args.p, args.nt))

    res = []
    for _run in range(args.runs):
        printr('run {}/{}'.format(_run + 1, args.runs))
        res.append(run(args.dataset, args.l, dim, args.nt, args))
        print('run {}/{} {}'.format(_run + 1, args.runs, res[-1]))

    np.savetxt('results/t_{}_{}_{}_{}'.format(
        args.dataset, args.l, args.p, args.nt), res)

    printr('')
    print('result {:.3f} ({:.3f})'.format(np.mean(res), np.std(res)))


if __name__ == '__main__':
    main()
