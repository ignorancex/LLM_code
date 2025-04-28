from coordsystem import *

class Coordinate(int):
    def __new__(cls, bs: Basis, value: int = 0):
        if value < 0:
            raise ValueError("A Coordinate cannot have a negative value")
        if bs == Basis.One and value < 1:
            raise ValueError("A OneBased Coordinate cannot have a value < 1")
        i = int.__new__(cls, value)
        i.bs = bs
        return i

    def to(self, tobs: Basis):
        assert(type(tobs) == Basis)
        if tobs == self.bs:
            return Coordinate(self.bs, self)
        elif tobs == Basis.Zero:
            return Coordinate(Basis.Zero, self - 1 )
        else:
            return Coordinate(Basis.Zero, self + 1)

        

