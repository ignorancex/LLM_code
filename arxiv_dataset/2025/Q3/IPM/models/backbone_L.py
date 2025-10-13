import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from .hourglass_L import HourglassNetGuid, HourglassNetWoSpp4qt



@torch.no_grad()
def flow_norm(tensorFlow):
    Backward_tensorGrid_cpu = {}
    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
    Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cpu()

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorFlow.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorFlow.size(2) - 1.0) / 2.0)], 1)
    return tensorFlow


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(ResidualBlock, self).__init__()
        if norm:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel),
                                      torch.nn.InstanceNorm2d(outchannel), nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel)
                                      torch.nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  # nn.BatchNorm2d(outchannel)
                                              torch.nn.InstanceNorm2d(outchannel))
        else:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel),
                                      # torch.nn.InstanceNorm2d(outchannel),
                                      nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel)
                                      # torch.nn.InstanceNorm2d(outchannel)
                                      )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  # nn.BatchNorm2d(outchannel)
                                              # torch.nn.InstanceNorm2d(outchannel)
                                              )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(UpResidualBlock, self).__init__()

        self.norm = norm
        # 计算上采样的输出通道数
        upsample_outchannel = outchannel * 2

        if norm:
            self.left = nn.Sequential(nn.Conv2d(inchannel, upsample_outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), torch.nn.InstanceNorm2d(upsample_outchannel), nn.ReLU(inplace=True), nn.ConvTranspose2d(upsample_outchannel, outchannel, kernel_size=2, stride=2, bias=False),
                                      torch.nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), torch.nn.InstanceNorm2d(outchannel))
        else:
            self.left = nn.Sequential(nn.Conv2d(inchannel, upsample_outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.ReLU(inplace=True), nn.ConvTranspose2d(upsample_outchannel, outchannel, kernel_size=2, stride=2, bias=False), )
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.left(x)
        shortcut = self.shortcut(x)

        # 使用 F.interpolate 进行上采样，实现宽高变为输入的两倍
        shortcut = F.interpolate(shortcut, scale_factor=2, mode='nearest')

        out += shortcut
        out = F.relu(out)
        return out


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # b = np.random.uniform(-1, 1)
        b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size()) * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g


class ACConv2d(nn.Module):
    """Asysmetirc kernels conv layers, refer to https://zhuanlan.zhihu.com/p/338800630

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ACConv2d, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.ac1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
        self.ac2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=bias)
        self.fusedconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        if self.training:
            ac1 = self.ac1(x)
            ac2 = self.ac2(x)
            x = self.conv(x)
            return (ac1 + ac2 + x) / 3
        else:
            x = self.fusedconv(x)
            return x

    def train(self, mode=True):
        super().train(mode=mode)
        if mode is False:
            weight = self.conv.weight.cpu().detach().numpy()
            weight[:, :, 1:2, :] = weight[:, :, 1:2, :] + self.ac1.weight.cpu().detach().numpy()
            weight[:, :, :, 1:2] = weight[:, :, :, 1:2] + self.ac2.weight.cpu().detach().numpy()
            self.fusedconv.weight = torch.nn.Parameter(torch.FloatTensor(weight / 3))
            if self.bias:
                bias = self.conv.bias.cpu().detach().numpy() + self.conv.ac1.cpu().detach().numpy() + self.conv.ac2.cpu().detach().numpy()
                self.fusedconv.bias = torch.nn.Parameter(torch.FloatTensor(bias / 3))
            if torch.cuda.is_available():
                self.fusedconv = self.fusedconv.cuda()


class AccResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(AccResidualBlock, self).__init__()
        if norm:
            self.left = nn.Sequential(ACConv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False),  # nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                                      # nn.BatchNorm2d(outchannel),
                                      nn.InstanceNorm2d(outchannel), nn.ReLU(inplace=True), ACConv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                                      # nn.BatchNorm2d(outchannel)
                                      nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(  # nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, bias=False),  # nn.BatchNorm2d(outchannel)
                    nn.InstanceNorm2d(outchannel), )
        else:
            self.left = nn.Sequential(  # nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel),
                # nn.InstanceNorm2d(outchannel),
                nn.ReLU(inplace=True), ACConv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(outchannel)
                # nn.InstanceNorm2d(outchannel)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, bias=False),  # nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                                              # nn.BatchNorm2d(outchannel)
                                              # nn.InstanceNorm2d(outchannel),
                                              )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texture_extractor = nn.Sequential(ACConv2d(1, 2, 5, 1, 2), ResidualBlock(2, 4, 3, 1, 2), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 2))
        self.res_extractor = nn.Sequential(ACConv2d(2, 8, 5, 1, 2), ResidualBlock(8, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 2))
        # self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 5, 1, 2), ResidualBlock(8, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 1), ResidualBlock(16, 16, 3, 1, 1))
        self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 5, 1, 2), ResidualBlock(8, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 2))

    def forward(self, text, res, mv):
        return torch.cat([self.texture_extractor(text), self.res_extractor(res), self.mv_extractor(mv)], dim=1)


class FeatureExtractor_light(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texture_extractor = nn.Sequential(ACConv2d(1, 2, 5, 1, 2), ResidualBlock(2, 4, 5, 2, 2), )
        self.res_extractor = nn.Sequential(ACConv2d(2, 8, 5, 1, 2), ResidualBlock(8, 16, 5, 2, 2), )
        self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 5, 1, 2), ResidualBlock(8, 16, 5, 2, 2), )

    def forward(self, text, res, mv):
        return torch.cat([self.texture_extractor(text), self.res_extractor(res), self.mv_extractor(mv)], dim=1)


class FeatureExtractor_light2mtt(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texture_extractor = nn.Sequential(UpResidualBlock(16, 8, 3, 1, 1, norm=True), ResidualBlock(8, 8, 3, 1, 1), )  # qt-net的纹理特征需要上采样2倍
        self.texture_extractor2 = nn.Sequential(ACConv2d(1, 4, 3, 1, 1), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 2))
        self.qt_extractor = nn.Sequential(UpResidualBlock(1, 4, 3, 1, 1, norm=True), UpResidualBlock(4, 8, 3, 1, 1), )  # 上采样两倍，减少模型的channels
        self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 3, 2, 1), ResidualBlock(8, 16, 3, 1, 2, norm=True), ResidualBlock(16, 16, 3, 1, 1))  # 输入为下采样的光流

    def forward(self, text, luma, qt_feat, mv):
        return torch.cat([self.texture_extractor(text), self.texture_extractor2(luma), self.qt_extractor(qt_feat), self.mv_extractor(mv)], dim=1)


class FeatureExtractor_light2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texture_extractor = nn.Sequential(  # ACConv2d(1, 2, 5, 1, 2),
            # ResidualBlock(2, 4, 5, 2, 2, norm=True),
            AccResidualBlock(1, 2, 3, 1, 1, norm=True), AccResidualBlock(2, 4, 3, 1, 2, norm=True), )
        self.angle_extractor = nn.Sequential(  # ACConv2d(4, 8, 5, 1, 2),
            # ResidualBlock(8, 16, 5, 2, 2, norm=True),
            AccResidualBlock(2, 4, 3, 1, 1, norm=True), AccResidualBlock(4, 8, 3, 1, 2, norm=True), )
        self.mag_extractor = nn.Sequential(  # ACConv2d(4, 8, 5, 1, 2),
            # ResidualBlock(8, 16, 5, 2, 2, norm=True),
            AccResidualBlock(2, 4, 3, 1, 1, norm=True), AccResidualBlock(4, 8, 3, 1, 2, norm=True), )

    def forward(self, text, mv_mag, mv_angle, mtt_mask):
        return torch.cat([self.texture_extractor(text) * mtt_mask, self.angle_extractor(mv_angle), self.mag_extractor(mv_mag)], dim=1)


class QT_Net_HLG(nn.Module):
    def __init__(self, qml=False, tml=False, guide=False) -> None:
        """_summary_

        Args:
            spp (bool, optional): spatial pyramid pooling. Defaults to False.
            qml (bool, optional): qp modualtion layer. Defaults to False.
            tml (bool, optional): TID Modulation Layer. Defaults to False.
        """
        super().__init__()
        self.guide = False
        self.feature_extractor = FeatureExtractor()
        if guide:
            self.guide = True
            self.guidance_extraction = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1), nn.ReLU())
            self.qt_net = HourglassNetGuid(nStacks=3, nModules=2, nFeat=64, nClasses=1, inplanes=48, qml=qml, tml=tml, guid_in_chans=8)
        else:
            self.qt_net = HourglassNetWoSpp4qt(nStacks=3, nModules=2, nFeat=64, nClasses=1, inplanes=48, qml=qml, tml=tml)  # self.mtt_dire_net = Luma_MttDirection_Net()

    def forward(self, luma, flow, qp=None, trans_flow_DAM=False, make_res=False, p0_frame=None, p1_frame=None, out_medium_feat=False, upsample=0, p0_flow=None, p1_flow=None):
        if trans_flow_DAM:
            # 将光流转换为其他格式
            with torch.no_grad():
                if flow is not None:
                    p0_flow, p1_flow = flow[:, 0], flow[:, 1]
                if upsample:
                    p0_flow = F.interpolate(p0_flow * upsample, scale_factor=upsample, mode='nearest')
                    p1_flow = F.interpolate(p1_flow * upsample, scale_factor=upsample, mode='nearest')
                p0_flow_norm, p1_flow_norm = flow_norm(p0_flow), flow_norm(p1_flow)
                p0_flow_angle, p1_flow_angle = torch.atan2(p0_flow_norm[:, 1], p0_flow_norm[:, 0]), torch.atan2(p1_flow_norm[:, 1], p1_flow_norm[:, 0])
                p0_flow_mag, p1_flow_mag = torch.norm(p0_flow_norm, dim=1), torch.norm(p1_flow_norm, dim=1)
                flow = torch.stack([p0_flow_mag, p1_flow_mag, p0_flow_angle, p1_flow_angle], dim=1)

        if make_res:
            with torch.no_grad():
                res = torch.cat([p0_frame - luma, p1_frame - luma], dim=1)

        res.requires_grad_(True)
        flow.requires_grad_(True)
        input_tensor = self.feature_extractor(luma, res, flow)
        if self.guide:
            # make up guidance based on size of res
            guidance = torch.ones_like(luma[:, 0:1])
            grid_size = 128
            for i in range(0, guidance.shape[-2], grid_size):
                guidance[:, :, i, :] = -1
            for j in range(0, guidance.shape[-1], grid_size):
                guidance[:, :, :, j] = -1
            guidance = self.guidance_extraction(guidance)
            if out_medium_feat:
                pred_qt, _, qt_feat = self.qt_net(input_tensor, qp=qp, guidance=guidance, out_mid=out_medium_feat)
            else:
                pred_qt, _ = self.qt_net(input_tensor, qp=qp, guidance=guidance)
        else:
            if out_medium_feat:
                pred_qt, _, qt_feat = self.qt_net(input_tensor, qp=qp, out_mid=out_medium_feat)
            else:
                pred_qt, _ = self.qt_net(input_tensor, qp=qp)
        if out_medium_feat:
            # 输出中间结果，减轻mtt-net的压力： texture feature, qt_feature
            return pred_qt, input_tensor[:, :16], qt_feat
        return pred_qt
