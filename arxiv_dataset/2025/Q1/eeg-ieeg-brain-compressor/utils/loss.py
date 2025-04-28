import torch
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


def reconstruction_loss(input, target):
    return F.l1_loss(input, target)


def line_loss(input, target):
    window_size = 128
    kernel = torch.ones([1, 1, window_size - 1], device=input.device)
    input_diff = torch.diff(input).abs()
    target_diff = torch.diff(target).abs()
    input_length = F.conv1d(input_diff, kernel, stride=(window_size // 2))
    target_length = F.conv1d(target_diff, kernel, stride=(window_size // 2))
    input_length = input_length[target_length != 0]
    target_length = target_length[target_length != 0]
    loss = F.l1_loss(input_length, target_length)
    return loss


def spectral_loss(input, target):
    device = input.device
    loss = 0
    for i in range(5, 12):
        n_fft = 2**i
        alpha = 1
        spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=n_fft // 4,
            normalized=True,  # Meta: True
            wkwargs={"device": device},
        ).to(device)
        input_spec = spectrogram(input)
        target_spec = spectrogram(target)
        l1 = (input_spec - target_spec).norm(p=1)
        l2 = (input_spec - target_spec).norm(p=2)
        n = input_spec.numel()
        loss += (l1 + alpha * l2) / n
    return loss


def feature_loss(fmaps_input, fmaps_target):
    loss = 0
    for fmap_input, fmap_target in zip(fmaps_input, fmaps_target):
        for feature_input, feature_target in zip(fmap_input, fmap_target):
            l1 = F.l1_loss(feature_input, feature_target)
            loss += l1 / feature_target.norm(p=1)
    return loss


def adversarial_loss(logits):
    loss = 0
    for logit in logits:
        loss += F.relu(1 - logit).mean()
    return loss


def discriminator_loss(real_logits, fake_logits):
    loss = 0
    for real_logit, fake_logit in zip(real_logits, fake_logits):
        loss += F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()
    return loss


def generator_loss(input, target, fmaps_input, fmaps_target, logits):
    return {
        "reconstruction_loss": reconstruction_loss(input, target),
        "line_loss": line_loss(input, target),
        "spectral_loss": spectral_loss(input, target),
        "feature_loss": feature_loss(fmaps_input, fmaps_target),
        "adversarial_loss": adversarial_loss(logits),
    }
