# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import tednet.tnn.tensor_ring as tr

import torch


class Test_TR:
    def test_TRConv2D(self):
        print('\n TRConv2D')
        data = torch.Tensor(16, 1, 28, 28)
        model = tr.TRConv2D([1], [4, 5], [6, 6, 6, 6], 3)
        res = model(data)

        assert len(res.shape) == 4

    def test_TRLinear(self):
        print('\n TRLinear')
        data = torch.Tensor(16, 20)
        model = tr.TRLinear([4, 5], [2, 5], [6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TRLeNet5(self):
        print('\n TRLeNet5')
        data = torch.Tensor(16, 1, 28, 28)
        model = tr.TRLeNet5(10, rs=[6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TRResNet20(self):
        print('\n TRResNet20')
        data = torch.Tensor(16, 3, 28, 28)
        model = tr.TRResNet20([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TRResNet32(self):
        print('\n TRResNet32')
        data = torch.Tensor(16, 3, 28, 28)
        model = tr.TRResNet32([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TRLSTM(self):
        print('\n TRLSTM')
        from collections import namedtuple
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        data = torch.Tensor(10, 16, 6)
        model = tr.TRLSTM([2, 3], [3, 3], [6, 6, 6, 6])

        state = LSTMState(torch.zeros(16, 9),
                          torch.zeros(16, 9))
        res, _ = model(data, state)

        assert len(res.shape) == 3


