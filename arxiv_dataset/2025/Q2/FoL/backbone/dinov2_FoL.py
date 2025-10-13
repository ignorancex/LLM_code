import torch
import torch.nn as nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch_dinov2_main/', model_name, source='local')
        self.num_channels = DINOV2_ARCHS[model_name]  # 768
        self.num_trainable_blocks = num_trainable_blocks  # 4
        self.norm_layer = norm_layer  # True
        self.return_token = return_token  # True


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        # Disabling gradient calculation for all but the last four blocks
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)  # Process the input through the first blocks without updating them
        x = x.detach()

        # Process through the last four blocks, enabling gradient calculation for them
        for i, blk in enumerate(self.model.blocks[-self.num_trainable_blocks:]):
            if i == 3:  # This is the third block from the end or the penultimate block in the entire sequence
                y = blk.norm1(x)
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                att_cls_to_patches = att[:, :, 0, 1:]
                att_cls_to_patches_mean = att_cls_to_patches.sum(dim=1)

            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)  # [480, 768, 16, 16]
        if self.return_token:
            return f, t, att_cls_to_patches_mean
        return f