#!/usr/bin/env python3

import pickle

from scipy import stats

kcp_ws_acc1_latency = pickle.load(open('./depth/kcp_ws_1/acc1_latency.pickle', 'rb'))
kcp_ws_acc2_latency = pickle.load(open('./depth/kcp_ws_2/acc1_latency.pickle', 'rb'))

print(kcp_ws_acc1_latency)
print(kcp_ws_acc2_latency)