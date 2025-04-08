
class Filter(object):
    def __init__(self, dataFactory, run_on_test = True, **args):
        super().__init__()
        self.dataFactory = dataFactory
        self._run_on_test = run_on_test

    def __call__(self, element, test=False):
        if not test or self._run_on_test:
            return self.apply(element)
        return element

    def prepare(self, first, others):
        return None

    def apply_one(self,  element, i, prepared):
        return element

    def apply_many(self, elements):
        prepared = self.prepare(elements[0], elements[1:])
        return tuple(self.apply_one(element, i, prepared) for i,element in enumerate(elements))

    def apply(self, element):
        if isinstance(element, tuple):
            return self.apply_many(element)
        else:
            return self.apply_many([element])[0]