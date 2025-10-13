import torch
import torch.nn as nn


def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)
    # Modify encoder with ARENA
    model.swinViT = SwinViTARENAWrapper(model.swinViT, rank=int(args.rank))


def set_bb(args):
    return []


def set_training_mode(model, args):
    model.eval()


def freeze_params(model, args):

    print("Freezing weights... ", end="\n")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    # Freeze/unfreeze decoder
    if args.adapt_hp["decoder"] != "frozen":
        print("UnFreezing decoder weights... ", end="\n")
        for name, param in model.named_parameters():
            if "decoder" in name:  # swin-unetr code for decoder
                param.requires_grad = True
            if args.model_id == "selfsup":
                if "encoder" in name:  # swin-unetr bottleneck
                    param.requires_grad = True


class ARENALayer(torch.nn.Module):
    def __init__(self, w, w_a, w_b, g):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.gating_vector = g

    def forward(self, x):
        a_out = self.w_a(x) # Compute A(x)

        # Convert g into a diagonal matrix and apply it using matrix multiplication
        g_diag = torch.diag(self.gating_vector)  # (dim, dim)
        gated_a = a_out @ g_diag  # Apply diagonal gating transformation

        x = self.w(x) + self.w_b(gated_a)  # ARENA adaptation
        return x


class SwinViTARENAWrapper(torch.nn.Module):
    def __init__(self, vit_model, rank=4):
        super(SwinViTARENAWrapper, self).__init__()
        self.ViTbase = vit_model
        self.rank = rank
        print('rank', self.rank)


        for i, layer in enumerate(list(self.ViTbase.children())[1:]):
            for ii, blocks in enumerate(layer.children()):
                for iii, block in enumerate(list(blocks.children())[0]):
                    w_a_linear_qkv = torch.nn.Linear(block.attn.qkv.in_features, self.rank*3, bias=False)
                    w_b_linear_qkv = torch.nn.Linear(self.rank*3, block.attn.qkv.out_features, bias=False)
                    torch.nn.init.zeros_(w_b_linear_qkv.weight)
                    g_qkv = nn.Parameter(torch.empty(self.rank*3).uniform_(-1, 1), requires_grad=True)
                    block.attn.qkv = ARENALayer(block.attn.qkv, w_a_linear_qkv, w_b_linear_qkv, g_qkv)

    def forward(self, x, mask_matrix):
        return self.ViTbase(x, mask_matrix)
