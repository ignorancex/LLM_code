from coordsystem import Basis
import unittest
from coordinate import Coordinate 

class TestCoordinate(unittest.TestCase):

    def test_conversions(self):
        coord = Coordinate(Basis.Zero, 1)
        self.assertEqual(Coordinate(Basis.One, 2), coord.to(Basis.One))
    
    def test_use_as_int(self):
        coord = Coordinate(Basis.Zero, 1)
        self.assertEqual(coord, 1)
        self.assertEqual(coord.to(Basis.One), 2)