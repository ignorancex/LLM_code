import pickle, os, sys, logging
import numpy as np
import torch, torchvision.transforms as transforms


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info("Arguments")
    for arg in vars(args):
        logger.info("    {:<22}        {}".format(arg+":", getattr(args,arg)) )
    logger.info("")

    return logger


@torch.inference_mode()
def evaluate(model, criterion, loader, device, encoder=None):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        if encoder is not None:
            x = encoder(x).to(device)
            y = y.to(device)
        else:
            x, y = x.to(device), y.to(device)

        _y = model(x)
        ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
        lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return {"acc": acc.average(), "loss": loss.average()}


__optim_zoo__ = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

def get_optim(name, parameters, **kwargs):
    return __optim_zoo__[name](parameters, **kwargs)


__scheduler_zoo__ = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
}

def get_scheduler(name, optimizer, **kwargs):
    return __scheduler_zoo__[name](optimizer, **kwargs)
