import torch
import numpy as np


def transformer(U, theta, **kwargs):

    def _meshgrid(height, width):

        x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))

        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(theta, input_dim):
        #num_batch, height, width = input_dim#.size()
        num_batch = input_dim[0]
        height = input_dim[1]
        width = input_dim[2]
        #  Changed
        theta = theta.reshape([-1, 3, 3]).float()

        # out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(height, width)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch,shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(theta, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]
        t_s = T_g[:,2,:]

        t_s_flat = t_s.reshape([-1])

        # smaller
        small = 1e-7
        smallers = 1e-6*(1.0 - torch.ge(torch.abs(t_s_flat), small).float())

        t_s_flat = t_s_flat + smallers
        #condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # Ty changed
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        flow_w = ((x_s_flat - grid[:,0,:].reshape([-1])) * (width/2)).reshape(num_batch, height, width)   # bs, h, w
        flow_h = ((y_s_flat - grid[:,1,:].reshape([-1])) * (height/2)).reshape(num_batch, height, width)
        flow = torch.stack([flow_w, flow_h], 3) # bs, h, w, 2
        flow = flow.permute(0,3,1,2)    # bs, 2, h, w
        return flow


    output = _transform(theta, U)
    return output#, condition