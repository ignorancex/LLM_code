import re
from xoa.commons import *


def unpack_args(str):
    if len(str) > 1:
        eq_re = re.compile("\s*=\s*")
        return dict(map(lambda x: eq_re.split(x),
                        re.compile("\s*,\s*").split(str)))
    else:
        return {}


def load_chooser(space, model_type, arg_string, verifier=None):
    
    args = unpack_args(arg_string)
    #debug("args of {} chooser: {}".format(model_type, args))

    if model_type == 'WGP':
        from .wgp import InputWarpedGPChooser
        return InputWarpedGPChooser(space, **args)
    elif model_type == 'RF':
        from .rf import RFChooser    
        return RFChooser(space, **args)
    else:
        from .base import BaseChooser
        return BaseChooser(space, **args)

   
