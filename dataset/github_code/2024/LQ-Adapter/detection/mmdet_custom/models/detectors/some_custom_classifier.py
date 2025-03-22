import torch.nn as nn
from torchvision import models
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from collections import OrderedDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv.runner import get_dist_info
import os
from mmcv_custom import load_checkpoint
# from focal_loss.focal_loss import FocalLoss


@DETECTORS.register_module()
class MyOwnResnet(nn.Module):
    def __init__(self, num_classes, backbone, backbone_ckpt = None, neck=None, pretrained=False, train_file = None, test_file = None, merged_file = None, *args, **kwargs):
        super(MyOwnResnet, self).__init__()
        self.merged_file = merged_file
        self.backbone_ckpt = backbone_ckpt
        
        self.m = torch.nn.Softmax(dim=-1)

        self.num_classes = num_classes
        # self.backbone = DDP(build_backbone(backbone), device_ids=[rank])
        self.backbone = build_backbone(backbone)
        load_checkpoint(self.backbone, self.backbone_ckpt)
        
        for p in self.backbone.parameters():
            # print(p)
            p.requires_grad_ = False
                
        self.annotations = self.file_to_annotations()
        # if neck is not None:
        #     self.neck = build_neck(neck)
            
        # self.neck = neck
        # inplanes, block, basewidth, scale = 1, GSoP_mode = 1, num_classes=1
        # model = 
        self.classifier = GbcNet(num_classes=num_classes)
        # self.loss = FocalLoss(gamma=0.65)
        self.loss = nn.CrossEntropyLoss()
        # self.classifier = models.resnet50(pretrained=True)
        # self.classifier.conv1 = nn.Conv2d(768,64,kernel_size=(3,3),stride=(2,2),padding=(3,3),bias=False)
        # num_ftrs = self.classifier.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        # self.classifier.fc = nn.Linear(num_ftrs, 2)
        
        # net = resnet50(pretrained=False) # You should put 'True' here 
        # self.classifier.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        # net(batch).size()
        
        # self.init_weights(pretrained=pretrained)
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        return x
    
    def file_to_annotations(self):
        content = {}
        with open(self.merged_file, 'r') as infile:
            for line in infile:
                c = line.strip().split(",")
                if c[1] == '1':
                    c[1] = 0
                elif c[1] == '2':
                    c[1] = 1
                else:
                    assert c[1] == '0'
                content.update({c[0]: int(c[1])})
        return content
            
        
    def file_name_to_label(file_name, annotation_file):
        pass
        
    def init_weights(self):
        pass
    def train_step(self, data, optimizer):
        # losses = self(**data)
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses, optimizer)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        
        # optimizer.step()
        return outputs
    def _parse_losses(self, losses, optimizer):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
                # loss_value.backward()
                # optimizer.step()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        
        losses = dict()
        outTensor = self.classifier(x)
        labels = torch.Tensor([self.annotations[t['ori_filename']] for t in img_metas]).to(outTensor.device)
        labels = labels.long()
        
        cross_entropy_loss = self.loss(self.m(outTensor), labels)
        
        losses.update(loss = cross_entropy_loss)

        return losses
    
    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""
    #     x = self.extract_feat(img)
    #     # losses = dict()
    #     outTensor = self.classifier(x, outTensor)
    #     # losses.update(loss)
    #     return outTensor
    
    def forward_test(self,
                      imgs,
                      img_metas,
                      gt_bboxes_ignore=None,
                      **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        x = self.extract_feat(imgs[0])
        outTensor = self.classifier(x)
        return torch.max(outTensor, 1)[1]
        
        
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


# class FocalLoss(nn.Module):
    
#     def __init__(self, weight=None, 
#                  gamma=2., reduction='none'):
#         nn.Module.__init__(self)
#         self.weight = weight
#         self.gamma = gamma
#         self.reduction = reduction
        
#     def forward(self, input_tensor, target_tensor):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#             ((1 - prob) ** self.gamma) * log_prob, 
#             target_tensor, 
#             weight=self.weight,
#             reduction = self.reduction)

class GbcNet(nn.Module):

    def __init__(self, inplanes = 768, GSoP_mode = 1, num_classes=2):
        self.inplanes = inplanes
        self.GSoP_mode = GSoP_mode
        super(GbcNet, self).__init__()
        # self.basewidth = basewidth
        # self.scale = scale
        # self.conv1 = nn.Conv2d(inplanes, inplanes // 2, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn1 = nn.SyncBatchNorm(inplanes // 2)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        # self.conv2 = nn.Conv2d(inplanes // 2, inplanes // 4, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn2 = nn.SyncBatchNorm(inplanes // 4)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        # self.conv3 = nn.Conv2d(inplanes // 4, inplanes // 8, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn3 = nn.BatchNorm2d(inplanes // 8)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        
        # self.layer1 = self._make_layer(block, 64, layers[0], att_position=att_position[0], att_dim=att_dim)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_position=att_position[1], att_dim=att_dim)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_position=att_position[2], att_dim=att_dim)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_position=att_position[3], att_dim=att_dim)
        # if GSoP_mode == 1:
        self.avgpool1 = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d(14, stride=1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d(14, stride=1)
        self.avgpool3 = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d(14, stride=1)
        self.avgpool4 = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d(14, stride=1)
        
        # self.fc0 = nn.Linear(inplanes, inplanes // 2)
        # self.relu0 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(inplanes, num_classes)
        # self.sigmoid = nn.Sigmoid()
            #print("GSoP-Net1 generating...")
        # else :
        #     self.isqrt_dim = 256
        #     self.layer_reduce = nn.Conv2d(512 * block.expansion, self.isqrt_dim, kernel_size=1, stride=1, padding=0,
        #                                   bias=False)
        #     self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
        #     self.layer_reduce_relu = nn.ReLU(inplace=True)
        #     self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1) / 2), num_classes)
            #print("GSoP-Net2 generating...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, att_position=[1], att_dim=128):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.basewidth, \
    #                         self.scale, att_position[0], att_dim=att_dim, stype='stage'))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, stride=1, downsample=None, \
    #                         basewidth=self.basewidth, scale=self.scale, \
    #                         attention=att_position[i], att_dim=att_dim))
    #     return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool1(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.maxpool2(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # x = self.maxpool3(x)
        
        x1 = self.avgpool1(x[0])
        x2 = self.avgpool2(x[1])
        x3 = self.avgpool3(x[2])
        x4 = self.avgpool4(x[3])
        x = x1+x2+x3+x4
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = x.view(x.size(0), -1)
        # print("HHHHHHHHHHH", x.shape)
        # x = self.fc0(x)
        # x = self.relu0(x)
        x = self.fc(x)

        return x