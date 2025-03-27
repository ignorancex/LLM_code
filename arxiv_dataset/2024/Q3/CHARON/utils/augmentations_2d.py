import random
import numpy as np
import pickle
import torch
import torch.nn.functional as F


def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam


def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result
    

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y

def convert_to_H36M(x):
    y = np.zeros((*x.shape[:2], 17, 3))
    y[:,:,0,:] = x[:,:,0,:]
    y[:,:,1,:] = x[:,:,16,:]
    y[:,:,2,:] = x[:,:,17,:]
    y[:,:,3,:] = x[:,:,18,:]
    y[:,:,4,:] = x[:,:,12,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,14,:]
    y[:,:,7,:] = x[:,:,1,:]
    y[:,:,8,:] = x[:,:,20,:]
    y[:,:,9,:] = x[:,:,2,:]
    y[:,:,10,:] = x[:,:,3,:]
    y[:,:,11,:] = x[:,:,4,:]
    y[:,:,12,:] = x[:,:,5,:]
    y[:,:,13,:] = x[:,:,6,:]
    y[:,:,14,:] = x[:,:,8,:]
    y[:,:,15,:] = x[:,:,9,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y


class random_rotataion:
    def __init__(self, theta=0.3):
        self.theta = theta

    def __call__(self, data):
        if data.shape[-1] == 2:
            return self.random_rot_2d(data)
        elif data.shape[-1] == 3:
            return self.random_rot_3d(data)
        else:
            raise ValueError(f'Unsupportable type of data was provided! Only 2 or 3 channels poses are allowed, but there was provided {data.shape[-1]} instead.')

    def _rot_3d(self, rot):
        """
        rot: T,3
        """
        cos_r, sin_r = rot.cos(), rot.sin()  # T,3
        zeros = torch.zeros(rot.shape[0], 1)  # T,1
        ones = torch.ones(rot.shape[0], 1)  # T,1

        r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
        rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
        rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
        rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

        ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
        r2 = torch.stack((zeros, ones, zeros),dim=-1)
        ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
        ry = torch.cat((ry1, r2, ry3), dim = 1)

        rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
        r3 = torch.stack((zeros, zeros, ones),dim=-1)
        rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
        rz = torch.cat((rz1, rz2, r3), dim = 1)

        rot = rz.matmul(ry).matmul(rx)
        return rot

    def random_rot_3d(self, data_torch):
        """
        data_numpy: C,T,V,M
        """
        # data_torch = torch.from_numpy(data_numpy)
        C, T, V, M = data_torch.shape
        data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
        rot = torch.zeros(3).uniform_(-self.theta, self.theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = self._rot(rot)  # T,3,3
        data_torch = torch.matmul(rot, data_torch)
        data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
        
        return data_torch

    def _rot_2d(self, rot):
        """
        rot: T,3
        """
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = torch.zeros(rot.shape[0], 1)
        ones = torch.ones(rot.shape[0], 1)

        r1 = torch.stack((ones, zeros),dim=-1)
        rx2 = torch.stack((cos_r[:,0:1], sin_r[:,0:1]), dim=-1)
        rx = torch.cat((r1, rx2), dim = 1) 

        ry1 = torch.stack((-sin_r[:,1:2], cos_r[:,1:2]), dim=-1)
        r2 = torch.stack((zeros, ones),dim=-1)
        ry = torch.cat((ry1, r2), dim = 1)

        rot = ry.matmul(rx)
        return rot

    def random_rot_2d(self, data_torch):
        """
        data_numpy: C,T,V,M
        """
        # data_torch = torch.from_numpy(data_numpy)
        C, T, V, M = data_torch.shape
        data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)
        rot = torch.zeros(2).uniform_(-self.theta, self.theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = self._rot_2d(rot)
        data_torch = torch.matmul(rot, data_torch)
        data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

        return data_torch
    

class random_move:
    def __init__(self, 
                 angle_candidate=(-10., -5., 0., 5., 10.),
                 scale_candidate=(0.9, 0.95, 1.0, 1.05, 1.1),
                 transform_candidate=(-0.2, -0.1, 0.0, 0.1, 0.2),
                 move_time_candidate=[1]):
        self.angle_candidate = angle_candidate
        self.scale_candidate = scale_candidate
        self.transform_candidate = transform_candidate
        self.move_time_candidate = move_time_candidate    

    def __call__(self, data):
        # input: C,T,V,M
        C, T, V, M = data.shape
        move_time = random.choice(self.move_time_candidate)
        node = torch.arange(0, T, T * 1.0 / move_time, dtype=torch.int32)
        node = torch.cat([node, torch.IntTensor([T])])
        num_node = len(node)

        A = torch.tensor(self.angle_candidate, dtype=torch.float)[torch.randint(len(self.angle_candidate), (num_node,))]
        S = torch.tensor(self.scale_candidate, dtype=torch.float)[torch.randint(len(self.angle_candidate), (num_node,))]
        T_x = torch.tensor(self.transform_candidate, dtype=torch.float)[torch.randint(len(self.angle_candidate), (num_node,))]
        T_y = torch.tensor(self.transform_candidate, dtype=torch.float)[torch.randint(len(self.angle_candidate), (num_node,))]

        a = torch.zeros(T)
        s = torch.zeros(T)
        t_x = torch.zeros(T)
        t_y = torch.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = torch.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * torch.pi / 180
            s[node[i]:node[i + 1]] = torch.linspace(S[i], S[i + 1],
                                                node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = torch.linspace(T_x[i], T_x[i + 1],
                                                node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = torch.linspace(T_y[i], T_y[i + 1],
                                                node[i + 1] - node[i])

        theta = torch.stack([torch.stack([torch.cos(a) * s, -torch.sin(a) * s]),
                             torch.stack([torch.sin(a) * s, torch.cos(a) * s])])

        # perform transformation
        for i_frame in range(T):
            xy = data[0:2, i_frame, :, :]
            new_xy = torch.mm(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data
    

class to_tensor:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(torch.float32)
        else:
            raise TypeError(f'Wrong type provided! {type(data)}')