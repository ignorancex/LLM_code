from algs.gdscore import GdScore

def create_alg(alg_name, val_loader, device, args):
    alg_dict = {
        "gdscore": GdScore
    }
    model = alg_dict[alg_name](val_loader, device, args)
    return model
