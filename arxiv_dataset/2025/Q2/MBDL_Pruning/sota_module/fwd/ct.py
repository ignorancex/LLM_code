import os.path

import torch
from .pytorch_radon.radon import Radon, IRadon
from .pytorch_radon.filters import RampFilter
import numpy as np
from .utility import addwgn
from torch.utils.data import Dataset
import tqdm
from sota_module.utility import check_and_mkdir
import pickle


class CTForwardModel:

    def grad_theta(self, x, y, theta, img_size):

        with torch.inference_mode(mode=False):

            theta = theta.clone().requires_grad_()

            sino = self.fmult(x, theta, img_size)

            loss_theta = torch.nn.MSELoss(reduction='sum')(y.clone(), sino)

            theta_grad = torch.autograd.grad(
                loss_theta, theta
            )

        return theta_grad[0]

    def grad(self, x, y, theta, img_size):
        g = self.ftran(self.fmult(x, theta, img_size) - y, theta, img_size)

        return g

    def fmult(self, x, theta, img_size):
        assert x.dim() == 2
        x = x.unsqueeze(0).unsqueeze(0)

        r = Radon(img_size, theta, False, dtype=torch.float, device=x.device)
        sino = r(x)

        sino = sino.squeeze(0).squeeze(0)

        return sino

    def ftran(self, z, theta, img_size, use_filter=None):
        assert z.dim() == 2
        z = z.unsqueeze(0).unsqueeze(0)

        ir = IRadon(img_size, theta, False, dtype=torch.float, use_filter=use_filter, device=z.device)
        reco_torch = ir(z)

        reco_torch = reco_torch.squeeze(0).squeeze(0)

        return reco_torch

    def imaging(self, x, num_angles, input_snr, angle_sigma, img_size):

        device = x.device

        # generate angle array
        theta_ipt = np.linspace(0., 180, num_angles, endpoint=False)

        if angle_sigma > 0:
            angle_noise = np.random.normal(0, angle_sigma, num_angles)
            theta_gt = theta_ipt + angle_noise
        else:
            theta_gt = theta_ipt

        # convert to torch
        theta_ipt = torch.tensor(theta_ipt, dtype=torch.float, device=device)
        theta_gt = torch.tensor(theta_gt, dtype=torch.float, device=device)

        # forward project
        sino = self.fmult(x.to(device), theta=theta_gt, img_size=img_size)

        if input_snr > 0:
            # add white noise to the sinogram
            sino, _ = addwgn(sino, input_snr)

        # backward project
        recon_bp = self.ftran(sino, theta=theta_ipt, img_size=img_size, use_filter=None)
        reco_fbp = self.ftran(sino, theta=theta_ipt, img_size=img_size, use_filter=RampFilter())

        return sino, recon_bp, reco_fbp, theta_gt, theta_ipt


class ComputedTomography(Dataset):

    def __init__(
            self,
            groundtruth,
            noise_snr,
            num_angle,
            angle_sigma,
            cache_id=None
    ):

        self.noise_snr = noise_snr
        self.num_angle = num_angle
        self.angle_sigma = angle_sigma

        self.x = groundtruth['x']
        assert self.x.dtype == torch.float32

        self.num_x, self.img_size, _ = self.x.shape
        self.fwd = CTForwardModel()

        if cache_id is not None:
            root_path = '/opt/dataset/cache_deq_cal/'
            check_and_mkdir(root_path)

            file_name = root_path + '%s_CT_noise_snr%d_num_angle%d_angle_sigma%.2f.pl' % (
                cache_id, noise_snr, num_angle, angle_sigma)

            if not os.path.exists(file_name):
                print("Cannot find cached data in disk, starting generating and saving.")
                self.cache_data = self.caching_data()

                with open(file_name, 'wb') as f:
                    pickle.dump(self.cache_data, f)

            else:
                print("Found cached data in disk, loading it.")
                with open(file_name, 'rb') as f:
                    self.cache_data = pickle.load(f)

        else:
            print("Not to use cached data, noted that it will cause different results for different running.")
            self.cache_data = self.caching_data()

    def caching_data(self):
        l = []

        for item in tqdm.tqdm(range(len(self)), desc='caching data'):
            l.append(self.__getitem__helper(item=item))

        return l

    def __len__(self):
        return self.num_x

    def __getitem__(self, item):
        return self.cache_data[item]

    def __getitem__helper(self, item):

        x = self.x[item]
        x = x.cuda()

        y, x0_bp, x0_fbp, theta_gt, theta = self.fwd.imaging(x, self.num_angle, self.noise_snr, self.angle_sigma, self.img_size)

        x0_fbp, y, theta, x, theta_gt = [i.cpu() for i in [x0_fbp, y, theta, x, theta_gt]]

        return x0_fbp, theta, y, {'img_size': self.img_size}, x, theta_gt
        # x0, theta0, y, (extra variable for forward model), x_groundtruth, theta_groundtruth

# for ii in [x, x0_fbp, x0_bp, y, theta, x, theta_gt]:
#     print(torch.min(ii), torch.max(ii), ii.shape, ii.dtype)
#
# exit(1)

# x_hat = x0_fbp
# for i in range(200):
#     x_hat = x_hat - 0.001 * self.fwd.grad(x_hat, y, theta, self.img_size)
#
#     print(i, torch.norm(x_hat - x))
#
# from skimage.io import imsave
# imsave('/opt/experiment/x0_fbp.tiff', x0_fbp.squeeze().numpy())
# imsave('/opt/experiment/x_hat.tiff', x_hat.squeeze().numpy())
#
# exit(1)

# for i in [x0_bp, x0_fbp, y, theta, x, theta_gt]:
#     print(i.shape, i.dtype)
# print(self.fwd.grad(x, y, theta, self.img_size))
# print(self.fwd.grad_theta(x, y, theta, self.img_size))

# from skimage.transform import radon, iradon
# x = x.numpy()
# # theta = np.linspace(0., 180., max(x.shape), endpoint=False)
# sinogram = radon(x, theta=theta.numpy())
# reconstruction_fbp = iradon(sinogram, theta=theta.numpy(), filter_name='ramp')
# from skimage.io import imsave
# imsave('/opt/experiment/reconstruction_fbp.tiff', reconstruction_fbp)
#
# imsave('/opt/experiment/x0_bp.tiff', x0_bp.squeeze().numpy())
# imsave('/opt/experiment/x0.tiff', x0.squeeze().numpy())
# # imsave('/opt/experiment/x.tiff', x.squeeze().numpy())
# exit(0)
