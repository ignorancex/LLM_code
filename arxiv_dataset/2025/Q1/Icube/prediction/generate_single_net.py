import json
from tqdm import *

elec_data = range(0, 10887)
junc_data = range(10887, 10887 + 4825)
bs_data = range(10887 + 4825, 10887 + 4825 + 20229)
aoi_data = range(10887 + 4825 + 20229, 10887 + 4825 + 20229 + 10533)
elec = ''
junc = ''
bs = ''
aoi = ''
with open('./GMNN_data/net.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        [s, t, _] = line.split('\t')
        s = int(s)
        t = int(t)
        if s in elec_data and t in elec_data:
            elec += '{0}\t{1}\t1\n'.format(s, t)
        elif s in junc_data and t in junc_data:
            junc += '{0}\t{1}\t1\n'.format(s, t)
        elif s in bs_data and t in bs_data:
            bs += '{0}\t{1}\t1\n'.format(s, t)
        elif s in aoi_data and t in aoi_data:
            aoi += '{0}\t{1}\t1\n'.format(s, t)
with open('./GMNN_data/elec_net.txt', 'w') as f:
    f.write(elec)
with open('./GMNN_data/junc_net.txt', 'w') as f:
    f.write(junc)
with open('./GMNN_data/bs_net.txt', 'w') as f:
    f.write(bs)
