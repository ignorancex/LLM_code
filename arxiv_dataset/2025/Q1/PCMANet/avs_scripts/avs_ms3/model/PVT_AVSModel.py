import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvt import pvt_v2_b5
from model.ca import QuerySelectedCA
# import pdb


def vec2mat(x, h, w):
    if len(x.size()) == 2:
        x = x.reshape(-1, x.shape[-1], 1, 1)
    x = x.repeat(1, 1, h, w)
    return x


class FPN(nn.Module):
    def __init__(self, channel):
        super(FPN, self).__init__()
        self.conv = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, L, H):
        if H == None:
            L = L
        else:
            if L.size()[2:] != H.size()[2:]:
                # (BCHW) => (HW)
                H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
            L = L + H
        L = self.bn(self.conv(L))
        L = F.relu(L, inplace=True)
        return L


class GF(nn.Module):
    def __init__(self, channel):
        super(GF, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, L, H, G):
        if H != None:
            H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
            # G1 = F.interpolate(G, size=H.size()[2:], mode='bilinear')
            G2 = F.interpolate(G, size=L.size()[2:], mode='bilinear')
            L = L + self.seq1(H) + self.seq2(G2)
        else:
            G = F.interpolate(G, size=L.size()[2:], mode='bilinear')
            L = L + self.seq1(G)
        L = self.seq3(L)

        return L


class QSCA(nn.Module):
    def __init__(self, in_channel, out_channel, size):
        super(QSCA, self).__init__()

        self.size = size
        self.gap = nn.AdaptiveAvgPool2d(size)
        self.aud_fc = nn.Linear(in_channel, out_channel)

        self.attn = QuerySelectedCA(
            image_size=self.size,
            patch_size=1,
            depth=1,
            heads=8,
            dim_head=32,
            dropout=0,
            emb_dropout=0
        )

    def forward(self, x, a, mask=None):
        b, c, h, w = x.shape
        x = self.gap(x)
        a = self.aud_fc(a)
        o, a = self.attn(x, a, mask)
        o = F.interpolate(o, (h, w), mode='bilinear')

        return o, a


class AVGA(nn.Module):
    def __init__(self, channel, head):
        super(AVGA, self).__init__()
        self.head = head
        self.seq1 = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 1), groups=head),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.Linear(channel, channel // head),
            nn.ReLU()
        )

    def forward(self, x, a):
        b, c, h, w = x.shape
        x = self.seq1(x)
        a = self.fc(a)
        a = vec2mat(a, h, w)
        x_list = torch.chunk(x, chunks=self.head, dim=1)
        x_n2_list = [F.normalize(_, p=2, dim=1) for _ in x_list]
        a = F.normalize(a, p=2, dim=1)
        att = [_ * a for _ in x_n2_list]
        att = [_.sum(dim=1, keepdim=True) for _ in att]
        x = [p * q for p, q in zip(att, x_list)]
        x = torch.cat(x, dim=1)

        return self.seq2(x)


class PCMANet(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256, config=None):
        super(PCMANet, self).__init__()
        self.cfg = config

        self.encoder_backbone = pvt_v2_b5()
        self.conv4 = nn.Sequential(nn.Conv2d(512, channel, (3, 3), padding=1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(320, channel, (3, 3), padding=1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, channel, (3, 3), padding=1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, channel, (3, 3), padding=1), nn.BatchNorm2d(channel), nn.ReLU())
        self.aud_fc = nn.Linear(128, channel)

        head = 8
        self.avga4 = AVGA(channel, head)
        self.avga3 = AVGA(channel, head)
        self.avga2 = AVGA(channel, head)
        self.avga1 = AVGA(channel, head)

        self.qsca4 = QSCA(channel, channel, 7)
        self.qsca3 = QSCA(channel, channel, 7)
        self.qsca2 = QSCA(channel, channel, 14)
        self.qsca1 = QSCA(channel, channel, 28)

        # self.FPN4 = FPN(channel)
        # self.FPN3 = FPN(channel)
        # self.FPN2 = FPN(channel)
        # self.FPN1 = FPN(channel)

        self.gf4 = GF(channel)
        self.gf3 = GF(channel)
        self.gf2 = GF(channel)
        self.gf1 = GF(channel)

        self.threshold = 0.99

        self.output_conv1 = nn.Sequential(
            nn.Conv2d(channel, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1, (1, 1))
        )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(channel, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1, (1, 1))
        )
        self.output_conv3 = nn.Sequential(
            nn.Conv2d(channel, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1, (1, 1))
        )
        self.output_conv4 = nn.Sequential(
            nn.Conv2d(channel, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1, (1, 1))
        )

        if self.training:
            self.initialize_weights()

    def forward(self, x, audio):
        """
        T = 5
        :param x: [b*T, c, h, w]
        :param audio: [b*T, d]
        """
        BF, C, H, W = x.shape
        x1, x2, x3, x4 = self.encoder_backbone(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # shape for pvt-v2-b5
        # BF x  64 x 56 x 56
        # BF x 128 x 28 x 28
        # BF x 320 x 14 x 14
        # BF x 512 x  7 x  7

        conv1_feat = self.conv1(x1)  # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)  # BF x 256 x 28 x 28
        conv3_feat = self.conv3(x3)  # BF x 256 x 14 x 14
        conv4_feat = self.conv4(x4)  # BF x 256 x  7 x  7
        audio = self.aud_fc(audio)  # aud [BF, 256]

        o4 = self.avga4(conv4_feat, audio)
        o3 = self.avga3(conv3_feat, audio)
        o2 = self.avga2(conv2_feat, audio)
        o1 = self.avga1(conv1_feat, audio)

        g4, a4 = self.qsca4(o4, audio, None)
        o4 = self.gf4(o4, None, g4)
        out4 = self.output_conv4(o4)
        mask4 = torch.sigmoid(out4)
        mask4a = (mask4 <= self.threshold).float().view(mask4.shape)
        mask4b = (mask4 >= 1 - self.threshold).float().view(mask4.shape)
        mask4 = mask4a * mask4b

        g3, a3 = self.qsca3(o3, a4, mask4)
        o3 = self.gf3(o3, o4, g3)
        out3 = self.output_conv3(o3)
        mask3 = torch.sigmoid(out3)
        mask3a = (mask3 <= self.threshold).float().view(mask3.shape)
        mask3b = (mask3 >= 1 - self.threshold).float().view(mask3.shape)
        mask3 = mask3a * mask3b
        up_mask4 = F.interpolate(mask4, mask3.size()[2:], mode='nearest')
        mask3 = up_mask4 * mask3

        g2, a2 = self.qsca2(o2, a3, mask3)
        o2 = self.gf2(o2, o3, g2)
        out2 = self.output_conv2(o2)
        mask2 = torch.sigmoid(out2)
        mask2a = (mask2 <= self.threshold).float().view(mask2.shape)
        mask2b = (mask2 >= 1 - self.threshold).float().view(mask2.shape)
        mask2 = mask2a * mask2b
        up_mask3 = F.interpolate(mask3, mask2.size()[2:], mode='nearest')
        mask2 = up_mask3 * mask2

        g1, a1 = self.qsca1(o1, a2, mask2)
        o1 = self.gf1(o1, o2, g1)
        out1 = self.output_conv1(o1)

        per4 = mask4.sum() / mask4.numel()
        per3 = mask3.sum() / mask3.numel()
        per2 = mask2.sum() / mask2.numel()
        per = [per4, per3, per2]

        # o4 = self.FPN4(o4 + vec2mat(audio, 7, 7), None)
        # o3 = self.FPN3(o3 + vec2mat(audio, 14, 14), o4)
        # o2 = self.FPN2(o2 + vec2mat(audio, 28, 28), o3)
        # o1 = self.FPN1(o1 + vec2mat(audio, 56, 56), o2)

        return F.interpolate(out1, (H, W), mode='bilinear'), F.interpolate(out2, (H, W), mode='bilinear'), \
            F.interpolate(out3, (H, W), mode='bilinear'), F.interpolate(out4, (H, W), mode='bilinear'), per

    def initialize_weights(self):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k: v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        # pdb.set_trace()


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    model = PCMANet()
    # output = model(imgs)
    output = model(imgs, audio)
