import einops
import torch
import torch.nn as nn

from backbone.STTFormer import STTFormer
from backbone.utils.modules import Pos_Embed

decoder_CONFIGS = {'tiny': [[128, 64, 64], [64, 32, 32], [32, 32, 16]],
                   'base': [[256, 128, 64], [128, 64, 32], [64, 64, 16]]}


def random_masking(x, masking_ratio: float):
    N, C, T, V = x.shape

    len_keep = int(T * (1 - masking_ratio))

    noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

    # keep the first subset
    ids_keep, _ = torch.sort(ids_shuffle[:, :len_keep], dim=1)
    ids_shuffle[:, :len_keep] = ids_keep

    ids_restore = torch.argsort(ids_shuffle, dim=1)

    x_unmasked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).repeat(1, C, 1, V))

    return x_unmasked, ids_restore


class SklMAE(STTFormer):
    def __init__(self, masking_ratio: float, config: str = 'tiny', **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        self.mask_tokens = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1), requires_grad=True)
        self.masking_ratio = masking_ratio

        self.decoder_pes = Pos_Embed(self.out_channels, self.num_frames, self.num_joints)
        self.decoder = STTFormer(config=decoder_CONFIGS[config], **kwargs)

        self.output_map = nn.Conv2d(decoder_CONFIGS[config][-1][1], kwargs["num_channels"], 1)

    def forward(self, x):

        B = x.shape[0]

        two_skel_mask = x[..., 1].sum((1, 2, 3)) > 0

        if not two_skel_mask.any():
            x = x[..., 0]
        else:
            x = torch.cat([x[..., 0], x[two_skel_mask, ..., 1]], dim=0)

        x = einops.rearrange(x, 'n c (t p) v -> n c t (p v)', p=self.len_parts)

        x = self.input_map(x)

        x += self.pes()

        if self.training:
            x_unmasked, ids_restore = random_masking(x, self.masking_ratio)
        else:
            x_unmasked = x

        for _, block in enumerate(self.blocks):
            x_unmasked = block(x_unmasked)

        cls = einops.rearrange(x_unmasked, 'nm c t v -> nm (t v) c 1')
        cls = self.drop_out2d(cls)
        cls = einops.reduce(cls, 'nm tv c 1 -> nm c', 'mean')
        cls = self.drop_out(cls)

        second_skel = torch.zeros_like(cls[:B])
        if two_skel_mask.any():
            second_skel[two_skel_mask] = cls[B:]
        second_skel[~two_skel_mask] = self.skel_token.repeat((~two_skel_mask).sum().item(), 1)
        cls = cls[:B] + second_skel

        pred = self.classifier(cls)

        if not self.training:
            return pred

        # append mask tokens to sequence
        mask_tokens = self.mask_tokens.repeat(x_unmasked.shape[0], 1, ids_restore.shape[1] - x_unmasked.shape[2], x_unmasked.shape[3])
        x = torch.cat([x_unmasked, mask_tokens], dim=2)
        x = torch.gather(x, dim=2, index=ids_restore.unsqueeze(1).unsqueeze(-1).repeat(1, x.shape[1], 1, x.shape[3]))  # unshuffle

        x += self.decoder_pes()

        for _, block in enumerate(self.decoder.blocks):
            x = block(x)

        x = self.output_map(x)

        # Restore the shape as it was before the tuple division
        x = einops.rearrange(x, 'n c t (p v) -> n c (t p) v', p=self.len_parts)
        second_skel = torch.zeros_like(x[:B]).unsqueeze(-1)
        rec = x[:B].unsqueeze(-1)
        rec = torch.cat([rec, second_skel], dim=-1)
        if two_skel_mask.any():
            rec[two_skel_mask, ..., 1] = x[B:]

        return pred, rec


if __name__ == '__main__':
    model = SklMAE(masking_ratio=0.75, num_channels=3, num_joints=25,
                   num_frames=120, num_classes=60, config='tiny')

    pred, x = model(torch.randn(3, 3, 120, 25, 2))
    print(pred.shape, x.shape)
