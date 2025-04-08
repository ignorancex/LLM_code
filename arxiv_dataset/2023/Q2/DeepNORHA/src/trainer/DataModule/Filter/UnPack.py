from .Filter import Filter

def unpack(struct) -> tuple:
    if isinstance(struct, (list,tuple)):
        ret = tuple(e for s in struct for e in unpack(s))
        return ret
    return tuple([struct])

class UnPack(Filter):
    def __init__(self, **args):
        super().__init__(**args)

    def apply_many(self, elements):
        return unpack(elements)