from enum import Enum

class Basis(Enum):
    Zero = 0
    One = 1

class End(Enum):
    Half_Open = 0
    Closed = 1


class Coordsystem(Enum):
    ZBHO = 0,
    ZBC = 1,
    OBHO = 2,
    OBC = 3,

    def get_basis(self) -> Basis:
        if self.value[0] > 1:
            return Basis.One
        else:
            return Basis.Zero
    
    def get_end(self) -> End:
        if self.value[0] % 2 == 0:
            return End.Half_Open
        else:
            return End.Closed


