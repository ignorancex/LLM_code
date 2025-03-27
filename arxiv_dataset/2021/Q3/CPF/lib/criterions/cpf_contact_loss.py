import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from lib.utils.builder import LOSS
from .focal_loss import multiclass_focal_loss, sigmoid_focal_loss

REGION_FOCAL_LOSS_ALPHA = dict(
    fphab=[
        0.09948797, 0.02393614, 0.00958673, 0.03823604, 0.02974173, 0.01377778, 0.19825081, 0.0638811, 0.03167253,
        0.10425754, 0.08043329, 0.02713633, 0.0742536, 0.08478072, 0.03479149, 0.02330341, 0.0624728
    ],
    ho3d=[
        0.10156912, 0.02822553, 0.01931505, 0.04377314, 0.02218014, 0.01778827, 0.19733622, 0.03988811, 0.04025808,
        0.19438567, 0.03801038, 0.02245348, 0.07134447, 0.05416137, 0.02648691, 0.01691766, 0.06590642
    ],
)


@LOSS.register_module()
class CPFContactLoss(nn.Module):

    def __init__(self, cfg):
        super(CPFContactLoss, self).__init__()
        self.contact_lambda_vertex_contact = cfg.CONTACT_LAMBDA_VERTEX_CONTACT
        self.contact_lambda_contact_region = cfg.CONTACT_LAMBDA_CONTACT_REGION
        self.contact_lambda_anchor_elasti = cfg.CONTACT_LAMBDA_ANCHOR_ELASTI

        # for focal loss
        self.focal_loss_alpha = cfg.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.FOCAL_LOSS_GAMMA
        self.region_focal_loss_alpha = torch.Tensor(REGION_FOCAL_LOSS_ALPHA[cfg.REGION_FOCAL_TYPE])

    def forward(self, preds, gts, **kwargs):
        contact_losses = {}

        # * =============================== VERTEX CONTACT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vertex_contact = preds["vertex_contact"]
        gt_vertex_contact = gts["vertex_contact"].float()

        # ? 1. we need to filter out the points that lie outside the image
        contact_in_image_mask = preds["contact_in_image_mask"]  # TENSOR (B, N)

        # ? 2. also, we need to filter out the points introduced in collate
        verts_padding_mask = gts["obj_verts_padding_mask"].float()  # TENSOR (B, N)
        combined_mask = contact_in_image_mask * verts_padding_mask  # TENSOR (B, N)

        vertex_contact_loss = sigmoid_focal_loss(
            inputs=vertex_contact,
            targets=gt_vertex_contact,
            masks=combined_mask,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="mean",
        )

        contact_losses["vertex_contact"] = vertex_contact_loss

        # * =============================== CONTACT REGION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        contact_region = preds["contact_region"]  # TENSOR(B, N, C)

        n_regions = contact_region.shape[2]  # C: 17
        contact_region = contact_region.view((-1, n_regions))  # TENSOR (BxN, C)

        # ======== convert gt region idx to one-hot >>>>>>>>>>>>
        gt_contact_region_id = gts["contact_region_id"]  # TENSOR(B, N)
        gt_contact_region_id = gt_contact_region_id.view((-1, 1))  # TENSOR(BxN, 1)
        gt_contact_region_id = gt_contact_region_id.long()

        # TENSOR(BxN, C+1)
        # considering vertexs without contact as background
        # the background class has the largest index (eg. 17)
        gt_contact_region_one_hot_with_back = torch.FloatTensor(gt_contact_region_id.shape[0], n_regions + 1).zero_()
        gt_contact_region_one_hot_with_back = gt_contact_region_one_hot_with_back.to(gt_contact_region_id.device)
        gt_contact_region_one_hot_with_back = gt_contact_region_one_hot_with_back.scatter_(1, gt_contact_region_id, 1)

        gt_contact_region_one_hot = gt_contact_region_one_hot_with_back[:, :n_regions]  # TENSOR (BxN , C)
        gt_contact_region_one_hot = gt_contact_region_one_hot.to(gt_contact_region_id.device)

        # ============== construct the mask >>>>>>>>>>>>>>>>>
        # ? 1. we need to filter out the points that lie outside the image
        recov_contact_in_image_mask = preds["contact_in_image_mask"].view((-1, 1))  # TENSOR (BxN, 1)

        # ? 2. also, we need to filter out the points introduced in collate
        verts_padding_mask = gts["obj_verts_padding_mask"].float().view((-1, 1))  # TENSOR (BxN, 1)

        # ? 3. third, we need to filter out the non-contact points in gt
        contact_filtering_mask = (gt_contact_region_id != n_regions).float()  # TENSOR (BxN, 1)
        contact_filtering_mask = contact_filtering_mask.to(gt_contact_region_id.device)

        region_combined_mask = recov_contact_in_image_mask * verts_padding_mask * contact_filtering_mask  # (BxN, 1)

        region_focal_loss_alpha = self.region_focal_loss_alpha.to(gt_contact_region_id.device)
        contact_region_loss = multiclass_focal_loss(
            inputs=contact_region,
            targets=gt_contact_region_one_hot,
            masks=region_combined_mask,
            alpha=region_focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="mean",
        )
        contact_losses["contact_region"] = contact_region_loss

        # * =============================== ANCHOR ELASTI >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        anchor_elasti = preds["anchor_elasti"]  # TENSOR (B, N, 4)
        gt_anchor_elasti = gts["anchor_elasti"].float()  # TENSOR (B, N, 4)

        maximum_anchor = anchor_elasti.shape[2]  # 4

        anchor_elasti = anchor_elasti.view((-1, maximum_anchor))  # TENSOR (BxN, 4)
        gt_anchor_elasti = gt_anchor_elasti.view((-1, maximum_anchor))  # TENSOR (BxN, 4)

        # ============== construct the mask >>>>>>>>>>>>>>>>>
        # ? 1. filter out the points that lie outside the image
        # ? 2. filter out the points introduced in collate
        # ? 3. filter out the non-contact points in gt
        # ? NOTE: 1.2.3. is already done as:  region_combined_mask
        region_combined_mask = region_combined_mask.repeat(1, maximum_anchor)  # TENSOR (BxN, 4)

        # ? 4. filter out the unbalanced region-anchors by CONTACT_ANCHOR_PADDING_MASK
        anchor_padding_mask = gts["anchor_padding_mask"].float()  # TENSOR (B, N, 4)
        anchor_padding_mask = anchor_padding_mask.view((-1, maximum_anchor))  # TENSOR (BxN, 4)

        anchor_combined_mask = anchor_padding_mask * region_combined_mask  # TENSOR (BxN, 4)
        anchor_combined_mask.requires_grad_ = False

        if anchor_combined_mask.sum().detach().cpu().item() != 0:
            anchor_elasti_loss = F.binary_cross_entropy(input=anchor_elasti, target=gt_anchor_elasti,
                                                        reduction="none")  # TENSOR (BxN, 4)
            anchor_elasti_loss = (anchor_elasti_loss * anchor_combined_mask).sum()  # TENSOR (BxN, 4) -> SUM
            anchor_elasti_loss = anchor_elasti_loss / (anchor_combined_mask.sum())  # reduction mean
            contact_losses["anchor_elasti"] = anchor_elasti_loss
        else:
            contact_losses["anchor_elasti"] = torch.Tensor([0.0]).float().to(gt_anchor_elasti.device)

        contact_loss_total = self.contact_lambda_vertex_contact * vertex_contact_loss + \
                             self.contact_lambda_contact_region * contact_region_loss + \
                             self.contact_lambda_anchor_elasti * anchor_elasti_loss
        contact_losses["contact_loss_total"] = contact_loss_total
        return contact_loss_total, contact_losses
