from coordsystem import *
from interval import Interval 
import unittest

class TestInterval(unittest.TestCase):

    def test_conversions(self):
        coords = Interval(Coordsystem.ZBHO, 1, 3)
        self.assertEqual(Interval(Coordsystem.ZBC, 1, 2), coords.to(Coordsystem.ZBC))
        self.assertEqual(Interval(Coordsystem.OBC, 2, 3), coords.to(Coordsystem.OBC))
        self.assertEqual(Interval(Coordsystem.OBHO, 2, 4), coords.to(Coordsystem.OBHO))
    
    def test_use_as_int(self):
        coords = Interval(Coordsystem.ZBHO, 1, 3)
        self.assertEqual(coords.start, 1)
        self.assertEqual(coords.end, 3)
        self.assertEqual(coords.to(Coordsystem.OBC).start, 2)
        self.assertEqual(coords.to(Coordsystem.OBC).end, 3)