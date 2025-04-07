import time
import argparse
import torch
import torch.nn as nn
from mmseg.apis import init_model

from mmseg.models.utils import resize


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    last_conv = None
    last_conv_name = None

    try:
        for name, child in module.named_children():
            if isinstance(child,
                          (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                if last_conv is None:  # only fuse BN that is after Conv
                    continue
                fused_conv = _fuse_conv_bn(last_conv, child)
                module._modules[last_conv_name] = fused_conv
                # To reduce changes, set BN as Identity instead of deleting it.
                module._modules[name] = nn.Identity()
                last_conv = None
            elif isinstance(child, nn.Conv2d):
                last_conv = child
                last_conv_name = name
            else:
                fuse_conv_bn(child)
    except Exception as e:
        pass

    return module


def test_speed(config_file, device, deploy):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    input = torch.randn(1, 3, 1024, 2048).to(device)
    model = init_model(config_file, device=device)
    model.eval()
    if deploy:
        model.backbone.switch_to_deploy()
    model = fuse_conv_bn(model)

    iterations = None
    with torch.no_grad():
        for _ in range(10):
            resize(
                input=model(input),
                size=input.shape[2:],
                mode='bilinear', )

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    resize(
                        input=model(input),
                        size=input.shape[2:],
                        mode='bilinear', )
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            resize(
                input=model(input),
                size=input.shape[2:],
                mode='bilinear', )
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    return FPS


if __name__ == '__main__':
    device = 'cuda:0'
    deploy = True  # Only RDRNet and GCNet are available, others will result in an error.
    number = 2
    configs = [
        # './mmsegmentation/configs/icnet/icnet_r18-d8_4xb2-160k_cityscapes-832x832.py',
        # './mmsegmentation/configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py',
        # './mmsegmentation/configs/cgnet/cgnet_fcn_4xb8-60k_cityscapes-512x1024.py',
        # './mmsegmentation/configs/bisenetv1/bisenetv1_r18-d32-in1k-pre_4xb4-160k_cityscapes-1024x1024.py',
        # './mmsegmentation/configs/bisenetv2/bisenetv2_fcn_4xb8-160k_cityscapes-1024x1024.py',
        # './mmsegmentation/configs/stdc/stdc1_4xb12-80k_cityscapes-512x1024.py',
        # './mmsegmentation/configs/stdc/stdc2_4xb12-80k_cityscapes-512x1024.py',
        # './mmsegmentation/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py',
        # './mmsegmentation/configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py',
        # './mmsegmentation/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py',
        # './mmsegmentation/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py',
        # './mmsegmentation/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py',
        './gcnet-s_4xb3-120k_cityscapes-1024x1024.py',
        './gcnet-m_4xb3-120k_cityscapes-1024x1024.py',
        './gcnet-l_4xb3-120k_cityscapes-1024x1024.py',
    ]

    for config in configs:
        print(f'========={config} Speed Testing=========')
        fps = 0
        for i in range(number):
            fps = fps + test_speed(config, device, deploy)
        fps = fps / number
        print(f"fps:{fps}")
