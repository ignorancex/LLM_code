from builtins import breakpoint
import torch
import copy
from collections import OrderedDict

lidar_pointpillar = 'ckpt/pillarnet_fullset_lidar.pth'
lidar_pillar_ckpt = torch.load(lidar_pointpillar)

new_state = dict()
new_state['epoch'] = lidar_pillar_ckpt['epoch']
new_state['it'] = lidar_pillar_ckpt['it']
new_state['optimizer_state'] = lidar_pillar_ckpt['optimizer_state']
new_state['version'] = lidar_pillar_ckpt['version']

new_state['model_state'] = OrderedDict()

for l2l_key, value in lidar_pillar_ckpt['model_state'].items():
    new_state['model_state'][l2l_key] = value
    new_l2l_key = 'radar_' + l2l_key
    new_state['model_state'][new_l2l_key] = value

torch.save(new_state, 'ckpt/pillarnet_fullset_init.pth')