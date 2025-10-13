import torch
import torch.nn as nn
import loralib


def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)
    # Modify encoder with AdaLoRA LoRA
    model.swinViT = SwinViTAdaLoRAWrapper(model.swinViT, rank=int(args.rank))


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
    

class SwinViTAdaLoRAWrapper(torch.nn.Module):
    def __init__(self, vit_model, rank=12):
        super(SwinViTAdaLoRAWrapper, self).__init__()
        self.ViTbase = vit_model
        self.rank = rank
        print('initial AdaLoRA rank', self.rank)


        for i, layer in enumerate(list(self.ViTbase.children())[1:]):
            for ii, blocks in enumerate(layer.children()):
                for iii, block in enumerate(list(blocks.children())[0]):

                    new_block_attn_qkv = loralib.SVDLinear(block.attn.qkv.in_features, block.attn.qkv.out_features, r=self.rank) 

                    with torch.no_grad():
                        new_block_attn_qkv.weight.copy_(block.attn.qkv.weight)
                        if block.attn.qkv.bias is not None and new_block_attn_qkv.bias is not None:
                            new_block_attn_qkv.bias.copy_(block.attn.qkv.bias)

                    block.attn.qkv = new_block_attn_qkv
                    block.attn.qkv.bias.requires_grad = False


    def forward(self, x, mask_matrix):
        return self.ViTbase(x, mask_matrix)
