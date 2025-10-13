import torch.nn as nn
import torch
from .CA import SELayerModulation
from .guidance import GuidanceBlock
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False, extra_padding=False):
        # extra_padding==True, 其中一个padding=0
        super(ResidualBlock, self).__init__()
        if norm:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding if not extra_padding else 0, bias=False),  # nn.BatchNorm2d(outchannel),
                                      torch.nn.InstanceNorm2d(outchannel), nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel)
                                      torch.nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  # nn.BatchNorm2d(outchannel)
                                              torch.nn.InstanceNorm2d(outchannel))
            elif padding == 0:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, bias=False, padding=padding),  # nn.BatchNorm2d(outchannel)
                                              torch.nn.InstanceNorm2d(outchannel))
        else:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=0 if extra_padding else padding, bias=False),  # nn.BatchNorm2d(outchannel),
                                      # torch.nn.InstanceNorm2d(outchannel),
                                      nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel)
                                      # torch.nn.InstanceNorm2d(outchannel)
                                      )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False, padding=padding),  # nn.BatchNorm2d(outchannel)
                                              # torch.nn.InstanceNorm2d(outchannel)
                                              )
            elif extra_padding:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, bias=False, padding=0),  # nn.BatchNorm2d(outchannel)
                                              # torch.nn.InstanceNorm2d(outchannel)
                                              )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HgResBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, padding=1):
        super(HgResBlock, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.padding = padding

        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=stride)
        self.bn_2 = nn.BatchNorm2d(midplanes)
        self.conv_2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=1, padding=padding)
        self.bn_3 = nn.BatchNorm2d(midplanes)
        self.conv_3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, stride=1)
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        if inplanes != outplanes and padding != 0:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
        elif padding == 0:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)

    # Bottle neck
    def forward(self, x):
        residual = x

        out = self.bn_1(x)
        out = self.conv_1(out)
        out = self.relu(out)

        out = self.bn_2(out)
        out = self.conv_2(out)
        out = self.relu(out)

        out = self.bn_3(out)
        out = self.conv_3(out)
        out = self.relu(out)

        if self.inplanes != self.outplanes or self.padding == 0:
            residual = self.conv_skip(residual)
        out += residual

        return out

class Hourglass(nn.Module):

    def __init__(self, depth, nFeat, nModules, resBlocks):
        super(Hourglass, self).__init__()

        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules
        self.resBlocks = resBlocks

        self.hg = self._make_hourglass()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlocks(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_hourglass(self):
        hg = []

        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))  # extra one for the middle
            hg.append(nn.ModuleList(res))

        return nn.ModuleList(hg)

    def _hourglass_forward(self, depth_id, x):
        up_1 = self.hg[depth_id][0](x)
        low_1 = self.downsample(x)
        low_1 = self.hg[depth_id][1](low_1)

        if depth_id == (self.depth - 1):
            low_2 = self.hg[depth_id][3](low_1)
        else:
            low_2 = self._hourglass_forward(depth_id + 1, low_1)

        low_3 = self.hg[depth_id][2](low_2)
        up_2 = self.upsample(low_3)

        return up_1 + up_2

    def forward(self, x):
        return self._hourglass_forward(0, x)


class HourglassNetGuid(nn.Module):
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3, qml=False, tml=False, guid_in_chans=None, no_pooling=False, guide_stride=2):
        """hourglassNet,reference https://www.cnblogs.com/xxxxxxxxx/p/11651437.html

        Args:
            nStacks (_type_): num of hourglass block, 3 by default
            nModules (_type_): num of resblocks in a single hourglass block
            nFeat (_type_): num of channels of residual block
            nClasses (_type_): channels of score layer
            resBlock (_type_, optional): _description_. Defaults to HgResBlock.
            inplanes (int, optional): _description_. Defaults to 3.
        """
        super(HourglassNetGuid, self).__init__()

        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.is_qml, self.is_tml = qml, tml
        # initial head
        self._make_head(no_pooling=no_pooling)
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        gud = []
        if self.is_qml:
            qml = []
        elif self.is_tml:
            tml = []

        for i in range(nStacks):
            hg.append(Hourglass(depth=3, nFeat=nFeat, nModules=nModules, resBlocks=resBlock))
            gud.append(GuidanceBlock(nFeat, nFeat, guid_in_chans, stride=guide_stride))
            if self.is_qml:
                qml.append(SELayerModulation(channel=nFeat, reduction=8))  # reduction > 1，减少FC参数
            elif self.is_tml:
                tml.append(SELayerModulation(channel=nFeat, reduction=8))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, kernel_size=1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, kernel_size=1))
                score_.append(nn.Conv2d(nClasses, nFeat, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        if self.is_qml:
            self.qml = nn.ModuleList(qml)
        elif self.is_tml:
            self.tml = nn.ModuleList(tml)

        self.gud = nn.ModuleList(gud)

    def _make_head(self, no_pooling):
        if no_pooling:
            self.head = nn.Sequential(self.resBlock(self.inplanes, self.nFeat), self.resBlock(self.nFeat, self.nFeat), self.resBlock(self.nFeat, self.nFeat))
        else:
            self.head = nn.Sequential(self.resBlock(self.inplanes, self.nFeat), nn.MaxPool2d(2, 2), self.resBlock(self.nFeat, self.nFeat), self.resBlock(self.nFeat, self.nFeat))

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1), nn.BatchNorm2d(outplanes), nn.ReLU())

    def forward(self, x, qp=None, tid=None, guidance=None, out_mid=False, mtt_0=False):
        """
        Args:
            x (tensor): _description_
            qp (float, optional): normalized qp, 0.5 + (qp / 51), shape (B,1). Defaults to None.
            tid (float, optional): tid of the current frame, including 0,1,2,3,4, shape (B,1). Defaults to None.
            out_mid: 是否输出qt-net的最后特征（conv压缩之前的）
            mtt_0: 如果当前模块是MTT划分的M1，则将中间两个module的结果也输出
        Returns:
            _type_: _description_
        """
        x = self.head(x)

        out = []
        if mtt_0:
            mid_feat = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.gud[i](y, guidance)
            if self.is_qml:
                y = self.qml[i](y, qp)
            elif self.is_tml:
                y = self.tml[i](y, tid)
            y = self.res[i](y)
            # 保存中间特征
            if out_mid and i == (self.nStacks - 1):
                out_feature = y
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < (self.nStacks - 1):
                if mtt_0:
                    mid_feat.append(y)
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        if out_mid:
            return out, y, out_feature
        elif mtt_0:
            return out, y, mid_feat
        else:
            return out, y

class HourglassNetWoSpp4qt(nn.Module):
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3, qml=False, tml=False, no_head=False):
        """hourglassNet,reference https://www.cnblogs.com/xxxxxxxxx/p/11651437.html

        Args:
            nStacks (_type_): num of hourglass block, 3 by default
            nModules (_type_): num of resblocks in a single hourglass block
            nFeat (_type_): num of channels of residual block
            nClasses (_type_): channels of score layer
            resBlock (_type_, optional): _description_. Defaults to HgResBlock.
            inplanes (int, optional): _description_. Defaults to 3.
            ds: 是否在score层增加下采样
        """
        super(HourglassNetWoSpp4qt, self).__init__()

        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.is_qml, self.is_tml = qml, tml

        # initial head
        if no_head:
            self.head = None
        else:
            self._make_head()
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        spp = []
        if self.is_qml:
            qml = []
        elif self.is_tml:
            tml = []

        for i in range(nStacks):
            hg.append(Hourglass(depth=3, nFeat=nFeat, nModules=nModules, resBlocks=resBlock))
            if self.is_qml:
                qml.append(SELayerModulation(channel=nFeat, reduction=8))  # reduction > 1，减少FC参数
            elif self.is_tml:
                tml.append(SELayerModulation(channel=nFeat, reduction=8))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, kernel_size=1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, kernel_size=1))
                score_.append(nn.Conv2d(nClasses, nFeat, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        if self.is_qml:
            self.qml = nn.ModuleList(qml)
        elif self.is_tml:
            self.tml = nn.ModuleList(tml)

    def _make_head(self):
        # self.conv_1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3)
        # self.bn_1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        # self.res_1 = self.resBlock(64, 128)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.res_2 = self.resBlock(128, 128)
        # self.res_3 = self.resBlock(128, self.nFeat)
        self.head = nn.Sequential(self.resBlock(self.inplanes, self.nFeat), nn.MaxPool2d(2, 2), self.resBlock(self.nFeat, self.nFeat), self.resBlock(self.nFeat, self.nFeat))

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1), nn.BatchNorm2d(outplanes), nn.ReLU())

    def forward(self, x, qp=None, tid=None, out_mid=False, mtt_0=False):
        if self.head:
            x = self.head(x)

        out = []
        if mtt_0:
            mid_feat = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            if self.is_qml:
                y = self.qml[i](y, qp)
            elif self.is_tml:
                y = self.tml[i](y, qp)
            y = self.res[i](y)
            if out_mid and i == (self.nStacks - 1):
                out_feature = y
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < (self.nStacks - 1):
                if mtt_0:
                    mid_feat.append(y)
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        if out_mid:
            return out, y, out_feature
        elif mtt_0:
            return out, y, mid_feat
        else:
            return out, y

