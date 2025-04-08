from .Filter import Filter

class Pack(Filter):
    def __init__(self, struct, fields=None, **args):
        super().__init__(**args)
        self._struct = struct
        self._fields = fields

    def apply_many(self, elements):
        if self._fields is not None:
            fields = { 
                field: element 
                for field, element in zip(self._fields, elements)
            }
            
        def pack(struct):
            if isinstance(struct, str):
                return fields[struct]
            if isinstance(struct, int):
                return elements[struct]
            return tuple(pack(substruct) for substruct in struct)

        return pack(self._struct)