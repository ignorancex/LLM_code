import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import dinov2_extract, sam2hiera
from fusion import CGAFusion, sff
from modules import updown, wtconv, RFB
from torchinfo import summary


class DGSUNet(nn.Module):
    def __init__(self,dino_model_name=None,dino_hub_dir=None,sam_config_file=None,sam_ckpt_path=None):
        super(DGSUNet, self).__init__()
        if dino_model_name is None:
            print("No model_name specified, using default")
            dino_model_name = 'dinov2_vitl14'
        if dino_hub_dir is None:
            print("No dino_hub_dir specified, using default")
            dino_hub_dir = 'facebookresearch/dinov2'
        if sam_config_file is None:
            print("No sam_config_file specified, using default")
            # Replace with your own SAM configuration file path
            sam_config_file = r'G:\MyProjectCode\SAM2DINO-Seg\sam2_configs\sam2.1_hiera_l.yaml'
        if sam_ckpt_path is None:
            print("No sam_ckpt_path specified, using default")
            # Replace with your own SAM pt file path
            sam_ckpt_path = r'G:\MyProjectCode\SAM2DINO-Seg\checkpoints\sam2.1_hiera_large.pt'
        # Backbone Feature Extractor
        self.backbone_dino = dinov2_extract.DinoV2FeatureExtractor(dino_model_name, dino_hub_dir)
        self.backbone_sam = sam2hiera.sam2hiera(sam_config_file,sam_ckpt_path)
        # Feature Fusion
        self.fusion4 = CGAFusion.CGAFusion(1152)
        # (1024,37,37)->(1024,11,11)
        self.dino2sam_down4 = updown.interpolate_upsample(11)
        # (1024,11,11)->(1152,11,11)
        self.dino2sam_down14 = wtconv.DepthwiseSeparableConvWithWTConv2d(in_channels=1024, out_channels=1152)
        self.rfb1 = RFB.RFB_modified(144, 64)
        self.rfb2 = RFB.RFB_modified(288, 64)
        self.rfb3 = RFB.RFB_modified(576, 64)
        self.rfb4 = RFB.RFB_modified(1152, 64)
        self.decoder1 = sff.SFF(64)
        self.decoder2 = sff.SFF(64)
        self.decoder3 = sff.SFF(64)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_dino, x_sam):
        # Backbone Feature Extractor
        x1, x2, x3, x4 = self.backbone_sam(x_sam)
        x_dino = self.backbone_dino(x_dino)
        # change dino feature map size and dimension
        x_dino4 = self.dino2sam_down4(x_dino)
        x_dino4 = self.dino2sam_down14(x_dino4)
        # Feature Fusion(sam & dino)
        x4 = self.fusion4(x4, x_dino4)
        # change fusion feature map dimension->(64,11/22/44/88,11/22/44/88)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.decoder1(x4,x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.decoder2(x,x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.decoder3(x,x1)
        out3 = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out1,out2,out3

######################################################################################################

if __name__ == "__main__":
    with torch.no_grad():
        model = DGSUNet().cuda()
        x_dino = torch.randn(1, 3, 518, 518).cuda()
        x_sam = torch.randn(1, 3, 352, 352).cuda()
        # print(model)
        summary(model, input_data=(x_dino, x_sam))
        out, out1, out2 = model(x_dino,x_sam)
        print(out.shape, out1.shape, out2.shape)

