from .Gaussian import Gaussian
from .IPM import IPM
from .ALittleIsEnough import ALittleIsEnough, ALittleIsEnough15
from .Mimic import Mimic
from .Min import MinMax, MinSum

def create_attack(args):
    if args.attack == 'gaussian':
        attack = Gaussian(args)
    elif args.attack == 'ipm':
        attack = IPM(args)
    elif args.attack == 'little':
        attack = ALittleIsEnough(args)
    elif args.attack == 'little15':
        attack = ALittleIsEnough15(args)
    elif args.attack == 'mimic':
        attack = Mimic(args)
    elif args.attack == 'minmax':
        attack = MinMax(args)
    elif args.attack == 'minsum':
        attack = MinSum(args)
    else:
        raise NotImplementedError('Unknown attack. ')

    return attack
