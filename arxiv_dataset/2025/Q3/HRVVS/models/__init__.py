from .vvs import VVS


def build_model(args):
    vvs = VVS(
        args.batch_size,
        vae_checkpoint='checkpoints/vae_ch160v4096z32.pth',
        var_checkpoint='checkpoints/var_d16.pth'
    )
    return vvs.cuda()