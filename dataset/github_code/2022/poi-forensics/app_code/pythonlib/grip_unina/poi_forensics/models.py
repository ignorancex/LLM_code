import os
import numpy as np
import tqdm
import torch
import segmentation_models_pytorch as smp


def Batch2Group(bn, num_groups):
    gn = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=bn.num_features,
        eps=bn.eps,
        affine=bn.affine).to(bn.weight.device)

    if gn.affine:
        # dem = torch.sqrt(bn.running_var+bn.eps)
        # gn.weight.data =  bn.weight / dem
        # gn.bias.data = bn.bias - bn.running_mean * gn.weight.data
        gn.weight.data = bn.weight.data
        gn.bias.data = bn.bias.data

    return gn


def convertAllBatch2Group(model, count = 0):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.BatchNorm2d):
            setattr(model, name, Batch2Group(layer, 4))
            count = count + 1
        else:
            count = count + convertAllBatch2Group(layer)[1]
    return model, count


class EncoderNet(torch.nn.Module):
    def __init__(self, encoder='resnet50', in_channels=257, out_channels=256, depth=5, weights='imagenet'):
        #weights='imagenet'
        super(EncoderNet, self).__init__()
        self.encoder = smp.encoders.get_encoder(encoder, in_channels=in_channels, depth=depth, weights=weights)
        self.encoder, conut_gn = convertAllBatch2Group(self.encoder)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.Linear(self.encoder._out_channels[-1], out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        elif len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.encoder(x)[-1]
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.last(x)[:, :self.out_channels]
        return x


class NanNet(torch.nn.Module):
    def __init__(self):
        super(NanNet, self).__init__()

    def forward(self, x):
        return torch.full_like(x, np.nan)


def load_model(resources_path, opt_model, device):
    weights_file = os.path.join(resources_path, opt_model['weights'])
    weights = torch.load(weights_file, map_location=device)

    if 'architecture_audio' in opt_model:
        assert opt_model['architecture_audio'].startswith('enc_')
        network_audio = EncoderNet(opt_model['architecture_audio'][4:],
                                   in_channels=1,
                                   out_channels=opt_model['feats_len'] * opt_model['factor_len']).to(device)

        if 'net_time' in weights:
            network_audio.load_state_dict(weights['net_time'])
        elif 'net_audio' in weights:
            network_audio.load_state_dict(weights['net_audio'])
        elif 'network' in weights:
            network_audio.load_state_dict(weights['network'])
        else:
            print(list(weights.keys()))
            network_audio.load_state_dict(weights)

        network_audio.out_channels = opt_model['feats_len']
    else:
        network_audio = None

    if 'architecture_video' in opt_model:
        assert opt_model['architecture_video'].startswith('enc_')
        network_video = EncoderNet(opt_model['architecture_video'][4:],
                                   in_channels=3 * opt_model['clip_length'] // opt_model['clip_video_stride'],
                                   out_channels=opt_model['feats_len'] * opt_model['factor_len']).to(device)

        if 'net_time' in weights:
            network_video.load_state_dict(weights['net_time'])
        elif 'net_video' in weights:
            network_video.load_state_dict(weights['net_video'])
        elif 'network' in weights:
            network_video.load_state_dict(weights['network'])
        else:
            print(list(weights.keys()))
            network_video.load_state_dict(weights)

        network_video.out_channels = opt_model['feats_len']
    else:
        network_video = None

    del weights
    return network_audio, network_video


def transfrom_video(frames):
    return torch.from_numpy(frames).permute((3, 0, 1, 2)).float()/256.0  # C x T x H x W
