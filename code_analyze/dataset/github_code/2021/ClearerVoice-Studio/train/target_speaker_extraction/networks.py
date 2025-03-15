import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args

        # audio backbone network
        if args.network_audio.backbone == 'seg':
            from models.seg.seg import seg
            self.sep_network = seg(args)
        elif args.network_audio.backbone == 'neuroheed':
            from models.neuroheed.neuroheed import neuroheed
            self.sep_network = neuroheed(args)
        elif args.network_audio.backbone == 'SpEx-plus':
            from models.SpEx_plus.SpEx_plus import SpEx_plus
            self.sep_network = SpEx_plus(args)
        elif args.network_audio.backbone == 'av_convtasnet':
            from models.av_convtasnet.av_convtasnet import av_convtasnet
            self.sep_network = av_convtasnet(args)
            self._define_lip_ref_encoder()
        elif args.network_audio.backbone == 'av_dprnn':
            from models.av_dprnn.av_dprnn import av_Dprnn
            self.sep_network = av_Dprnn(args)
            self._define_lip_ref_encoder()
        elif args.network_audio.backbone == 'av_tfgridnet':
            from models.av_tfgridnetV3.av_tfgridnetv3_separator import av_TFGridNetV3
            self.sep_network = av_TFGridNetV3(args)
            self._define_lip_ref_encoder()
        elif args.network_audio.backbone == 'av_mossformer2':
            from models.av_mossformer2.av_mossformer2 import av_Mossformer
            self.sep_network = av_Mossformer(args)
            self._define_lip_ref_encoder()
        else:
            raise NameError('Wrong network selection')

    def _define_lip_ref_encoder(self):
        # reference network for lip encoder
        assert self.args.network_reference.cue == 'lip'

        if self.args.network_reference.backbone == 'resnet18':
            from models.visual_frontend.resnet18 import Visual_encoder
        elif self.args.network_reference.backbone == 'blazenet64':
            from models.visual_frontend.blazenet64 import Visual_encoder
        else:
            raise NameError('Wrong reference network selection')
        self.ref_encoder = Visual_encoder(self.args)


    def forward(self, mixture, ref=None):
        if self.args.network_audio.backbone == 'seg':
            # gesture based speaker extraction
            return self.sep_network(mixture, ref)
        elif self.args.network_audio.backbone == 'neuroheed':
            # neuro-steered speaker extraction
            return self.sep_network(mixture, ref)
        elif self.args.network_audio.backbone == 'SpEx-plus':
            # audio-based speaker extraction
            aux, aux_len, speakers = ref
            aux = aux.to(self.args.device)
            aux_len = aux_len.to(self.args.device)#.unsqueeze(1)
            speakers = speakers.to(self.args.device)

            ests, ests2, ests3, spk_pred = self.sep_network(mixture, aux, aux_len)
            if torch.sum(speakers) >=0:
                return (ests, ests2, ests3, spk_pred, speakers)
            else: return ests
        elif self.args.network_audio.backbone in ['av_convtasnet', 'av_dprnn', 'av_tfgridnet', 'av_mossformer2']:
            # speaker extraction with lip reference
            if self.args.network_reference.cue == 'lip':
                ref = ref.to(self.args.device)
                ref = self.ref_encoder(ref)
                return self.sep_network(mixture, ref)
            else:
                raise NameError('Wrong network and reference combination selection')
        else:
            raise NameError('Wrong network selection')



