import unittest
import io
import fetch_afdb
import mock

class Testfetch_afdb(unittest.TestCase):
    """
    UnitTest for alphafold annotations
    """

    @mock.patch("builtins.open")
    def test_get_afdb_ids(self, file_mock):
        file_mock.return_value = io.StringIO(
            "A8H2R3,1,199,AF-A8H2R3-F1,2\n"
            "Q5EP08,1,153,AF-Q5EP08-F1,2\n"
            "Q9DDD7,1,292,AF-Q9DDD7-F1,2\n"
            "P60709,1,375,AF-P60709-F1,2\n")
        alphafold_ids = {'A8H2R3', 'Q5EP08', 'Q9DDD7', 'P60709'}
        self.assertEqual(fetch_afdb.get_afdb_ids(file_mock), (alphafold_ids))

