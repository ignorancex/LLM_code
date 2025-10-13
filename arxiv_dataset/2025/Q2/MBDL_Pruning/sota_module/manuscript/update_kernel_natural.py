from dataset.natural_image import NaturalImageDataset, grad_theta, G, load_kernel_via_idx
import torch
from tifffile import tifffile


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))

    return otf


def run(config):

    dataset = NaturalImageDataset(
        subset='set12_tst',
        root_path='/opt/dataset/natural_image',
        is_preload=False,
        noise_snr=50,
        kernel_idx=10,
        down_sampling_factor=1,
        cache_id="nips2022_beta"
    )

    x_gt, theta_gt, y = dataset[5]

    theta_input = theta_gt + torch.randn_like(theta_gt) * 0.0001
    # theta_input = load_kernel_via_idx(11).unsqueeze(0).unsqueeze(0)

    theta_hat = theta_input.clone().requires_grad_()
    optimizer = torch.optim.Adam(params=[theta_hat], lr=1e-8)
    loss = torch.nn.MSELoss()

    for i in range(1000):
        print(i, torch.norm(G(x_gt, theta_hat, sf=1) - y), torch.norm(theta_hat - theta_gt))

        # Fk = p2o(theta_gt, shape=(256, 256))
        # Fx = torch.fft.fftn(x_gt, dim=[-2, -1])
        # Fy = torch.fft.fftn(y, dim=[-2, -1])
        #
        # dc = torch.fft.ifftn(torch.conj(Fx) * Fx * Fk - torch.conj(Fx) * Fy, s=[25, 25], dim=[-2, -1]).real
        #
        # theta_hat = theta_hat + 1e-9 * dc

        predict = G(x_gt, theta_hat, sf=1)
        loss_theta = loss(y, predict)

        loss_theta.backward()
        optimizer.step()

    # exit(0)
    #
    # theta_hat = theta_input
    # for i in range(10000):
    #     print(i, torch.norm(G(x_gt, theta_hat, sf=1) - y), torch.norm(theta_hat - theta_gt))
    #     theta_hat = theta_hat - 1e-8 * grad_theta(x_gt, y, theta_hat, sf=1)

    tifffile.imwrite('/opt/experiment/theta_input.tiff', theta_input.detach().numpy())
    tifffile.imwrite('/opt/experiment/theta_hat.tiff', theta_hat.detach().numpy())

    tifffile.imwrite('/opt/experiment/theta_gt.tiff', theta_gt.detach().numpy())
