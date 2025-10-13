import torch


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y, noise

