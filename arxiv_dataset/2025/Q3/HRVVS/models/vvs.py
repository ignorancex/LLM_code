import torch
from itertools import chain
from .dualbranch_seg import MVANet
from .vqvae import VQVAE
from .var import VAR
from torch import nn


MEMORY_BANK = [[]]
LAST_PRED = [None]
loss_func_var = torch.nn.CrossEntropyLoss(reduction='none')



class VVS(nn.Module):
    def __init__(self, mem_length, vae_checkpoint='checkpoints/vae_ch160v4096z32.pth',
                 var_checkpoint='checkpoints/var_d16.pth'):
        super().__init__()
        self.patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        self.token_nums = [int(pn**2) for pn in self.patch_nums]
        self.mem_length = mem_length
        
        self.vae = VQVAE(
            vocab_size=4096, 
            z_channels=32,
            ch=160,
            share_quant_resi=4,
            v_patch_nums=self.patch_nums,
            test_mode=True
        )
        
        self.var = VAR(
            vae_local=self.vae,
            num_classes=1000,
            depth=16,
            embed_dim=16*64,
            num_heads=16,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1*16/24,
            norm_eps=1e-6,
            shared_aln=False,
            cond_drop_rate=0.1,
            attn_l2_norm=True,
            patch_nums=self.patch_nums,
            flash_if_available=True,
            fused_if_available=True
        )
        
        self.vae.load_state_dict(torch.load(vae_checkpoint, map_location='cpu', weights_only=True), strict=False)
        self.var.load_state_dict(torch.load(var_checkpoint, map_location='cpu', weights_only=True), strict=False)
        
        for name, param in chain(self.var.named_parameters(), self.vae.named_parameters()):
            param.requires_grad = 'adapter' in name
        self.seg_model = MVANet()

    
    def forward(self, x, vid):
        f = self.vae.quant_conv(self.vae.encoder(x))
        index_list = self.vae.quantize.f_to_idxBl_or_fhat(f, to_fhat=False)
        index_gt = torch.cat(index_list, dim=1)
        input = self.vae.quantize.idxBl_to_var_input(index_list)
        logits = self.var(torch.ones(len(x), dtype=torch.long, device='cuda'), input)
        var_loss = loss_func_var(logits.view(-1, self.vae.vocab_size), index_gt.view(-1)).view(1, -1).sum(dim=-1).mean()
        idx_BL = torch.argmax(logits, dim=-1)
        idx_BL = [tensor.long() for tensor in torch.split(idx_BL, self.token_nums, dim=1)]
        f_hats = self.vae.idxBl_to_f_hat(idx_BL, same_shape=False)
        
        f_for_seg = []
        f_size = f_hats[-1].shape[-1]
        for f_hat in reversed(f_hats):
            if f_hat.shape[-1] == f_size:
                f_for_seg.append(f_hat)
                f_size /= 2
        
        global MEMORY_BANK, LAST_PRED
        while len(MEMORY_BANK) <= vid:
            MEMORY_BANK.append([])
            LAST_PRED.append(None)

        if self.training:
            sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1, glb_e5, glb_e1 = self.seg_model.forward(x, f_for_seg, MEMORY_BANK[vid], LAST_PRED[vid])
        else:
            final, glb_e5, glb_e1 = self.seg_model.forward(x, f_for_seg, MEMORY_BANK[vid], LAST_PRED[vid])
        
        if len(MEMORY_BANK[vid]) >= self.mem_length:
            MEMORY_BANK[vid].clear()
        MEMORY_BANK[vid].append(glb_e5.detach())
        LAST_PRED[vid] = glb_e1.detach()
        
        if self.training:
            return sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1, var_loss / var_loss.detach()
        else:
            return final

    
    def cuda(self):
        self.seg_model = self.seg_model.cuda()
        self.vae = self.vae.cuda()
        self.var = self.var.cuda()
        return self 