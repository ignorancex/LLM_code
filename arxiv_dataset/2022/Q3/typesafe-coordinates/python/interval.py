from coordsystem import Coordsystem
from coordinate import *

class Interval:
    def __init__(self, cs: Coordsystem, start: int = 0, end: int = 1):
        if start < 0 or end < 0:
            raise ValueError("An Interval start or end cannot have a negative value")
        if cs.get_basis() == Basis.One and (start < 1 or end < 1):
            raise ValueError("A OneBased Interval cannot have a value < 1")
        self.cs = cs
        bs = cs.get_basis()
        self.start = Coordinate(bs, start)
        self.end = Coordinate(bs, end)

    def __str__(self):

        return "{0}b[{1},{2}{3}".format(
            0 if self.cs.get_basis() == Basis.Zero else 1, 
            self.start, 
            self.end, 
            "]" if self.cs.get_end() == End.Closed else ")"
        )
    
    def __eq__(self, other: object) -> bool:
        if self.cs != other.cs: return False
        if self.start != other.start: return False
        if self.end != other.end: return False
        return True

    def to(self, tocs: Coordsystem):
        assert(type(tocs) == Coordsystem)
        start = self.start.to(tocs.get_basis())
        end = self.end.to(tocs.get_basis())
        if tocs.get_end() == self.cs.get_end():
            return Interval(tocs, start, end)
        elif tocs.get_end() == End.Half_Open:
            return Interval(tocs, start, end + 1)
        else:
            return Interval(tocs, start, end - 1)