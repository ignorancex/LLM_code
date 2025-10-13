class TripletEnd:
    def __init__(self, name: str, node_type: str, is_constant: bool, candidates: set = None):
        self.name = name
        self.node_type = node_type
        self.is_constant = is_constant
        self.candidates = candidates
        self.properties = {}

    def __iter__(self):
        return self.candidates.__iter__()

    def __repr__(self):
        return f"{self.get_uid()}, {self.is_constant=}, {self.properties=}, {self.candidates=}"

    def __str__(self):
        return self.__repr__()

    def get_uid(self):
        return self.name + "::" + str(self.node_type)

    def intersection_update(self, node_ids):
        self.candidates = set(node_ids).intersection(self.candidates)


class Triplet:
    def __init__(self, h: TripletEnd, e: str, t: TripletEnd):
        self.h = h
        self.e = e
        self.t = t

    def __repr__(self):
        return f"{self.h.get_uid()} -> {self.e} -> {self.t.get_uid()}"

    def __str__(self):
        return self.__repr__()