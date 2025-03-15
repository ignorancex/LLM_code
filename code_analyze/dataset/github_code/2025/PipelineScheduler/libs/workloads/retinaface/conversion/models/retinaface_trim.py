import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        # self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        # ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1))
        return output


class ProcModel(nn.Module):
    def __init__(self, model, class_num, w, h, device):
        super(ProcModel, self).__init__()
        self.model = model
        self.class_num = class_num
        self.w = w
        self.h = h
        self.anchors = create_anchor(w, h).to(device)

    def forward(self, x):
        bboxes, scores = self.model(x)
        cx = self.anchors[:, 0] + bboxes[:, :, 0] * 0.1 * self.anchors[:, 2]
        cy = self.anchors[:, 1] + bboxes[:, :, 1] * 0.1 * self.anchors[:, 3]
        sx = self.anchors[:, 2] + torch.exp(bboxes[:, :, 2] * 0.2)
        sy = self.anchors[:, 3] + torch.exp(bboxes[:, :, 3] * 0.2)

        y1 = (cx - sx / 2) * self.w
        x1 = (cy - sy / 2) * self.h
        y2 = (cx - sx / 2) * self.w
        x2 = (cy - sy / 2) * self.h

        bboxes = torch.stack((x1, y1, x2, y2), dim=2)
        bboxes = torch.unsqueeze(bboxes, dim=2)

        scores = scores[:, :, 1:]
        return [bboxes, scores]

def create_anchor(h=288, w=320, stride=[8, 16, 32]):
    from math import ceil
    feature_map = []
    for i in range(0, len(stride)):
        feature_map.append([ceil(h / stride[i]), ceil(w / stride[i])])

    min_sizes = [
        [10, 20],
        [32, 64],
        [128, 256]
    ]
    anchors = []

    for k in range(len(feature_map)):
        min_size = min_sizes[k]
        for i in range(feature_map[k][0]):
            for j in range(feature_map[k][1]):
                for m in range(len(min_size)):
                    s_kx = min_size[m] * 1.0 / w
                    s_ky = min_size[m] * 1.0 / h
                    cx = (j + 0.5) * stride[k] / w
                    cy = (i + 0.5) * stride[k] / h
                    anchors.append([cx, cy, s_kx, s_ky])
    return torch.FloatTensor(anchors)
