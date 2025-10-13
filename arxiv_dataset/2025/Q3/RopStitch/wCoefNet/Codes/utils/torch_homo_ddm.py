import torch
import numpy as np


def transformer(U, DisPt, **kwargs):

    def _meshgrid(height, width):

        x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-0.5, 0.5, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-0.5, 0.5, height), 1),
                               torch.ones([1, width]))

        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(input_dim, DisPt):
        # DisPt: bs, 4
        # det_p: bs, 4, 2

        num_batch, num_channels , height, width = input_dim.size()

        grid = _meshgrid(height, width)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch,shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        coord = grid[:,0:2,:] # bs, 2, h*w
        x1 = torch.tensor([-0.5]).cuda().unsqueeze(0).expand(num_batch, -1)   #bs, 1
        x2 = torch.tensor([0.5]).cuda().unsqueeze(0).expand(num_batch, -1)   #bs, 1
        y1 = torch.tensor([0.5]).cuda().unsqueeze(0).expand(num_batch, -1)   #bs, 1
        y2 = torch.tensor([-0.5]).cuda().unsqueeze(0).expand(num_batch, -1)   #bs, 1

        wa = (x2-coord[:,0,:])*(coord[:,1,:]-y2)    # bs, h*w
        wb = (coord[:,0,:]-x1)*(coord[:,1,:]-y2)
        wc = (x2-coord[:,0,:])*(y1-coord[:,1,:])
        wd = (coord[:,0,:]-x1)*(y1-coord[:,1,:])
 

        ddm = wa*DisPt[:,2].unsqueeze(1) + wb*DisPt[:,3].unsqueeze(1) + wc*DisPt[:,0].unsqueeze(1) + wd*DisPt[:,1].unsqueeze(1)
        ddm_map = ddm.reshape([num_batch, 1, height, width])
        return ddm_map


    output = _transform(U, DisPt)
    return output#, condition