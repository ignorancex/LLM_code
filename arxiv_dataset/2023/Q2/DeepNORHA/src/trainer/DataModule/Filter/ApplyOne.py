from .Filter import Filter

class ApplyOne(Filter):
    """
    Applies a pipe to a single element of a tuple.

    Example:
        - Type: ApplyOne
          index: 0
          pipe:
          - ...
          - ...
    """
    def __init__(self, index, pipe, **args):
        super().__init__(**args)
        self._index = index
        self._pipe = self.dataFactory.getPipe(pipe)

    def apply_one(self, element, i, prepared):
        if i != self._index:
            return super().apply_one(element, i, prepared)

        for filter in self._pipe:
            element = filter(element)
        return element