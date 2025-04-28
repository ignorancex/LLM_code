import unittest
import resources.ComponentsMapping
from models import Ligand

class TestComponentsMapping(unittest.TestCase):
    """
    UnitTest for Ligand annotations
    """
    def setUp(self):
        super(TestComponentsMapping, self).setUp()
        self.ligands = [
            Ligand("EMD-8959", "2", HET="CA", name="CALCIUM ION", copies="4"),
            Ligand("EMD-8959", "3", HET="D12", name="DODECANE", copies="8"),
            Ligand("EMD-8959", "4", HET="D10", name="DECANE", copies="2")]
        self.chembl_map = {'D12': 'CHEMBL30959', 'D10': 'CHEMBL134537'}
        self.chebi_map = {'D12': '28817', 'CA': '29108'}
        self.drugbank_map = {'CA': 'DB14577'}


    def test_worker(self):
        LigandMap = resources.ComponentsMapping.ComponentsMapping(self.ligands)
        chembl = ["", "CHEMBL30959", "CHEMBL134537"]
        chebi = ["29108", "28817", ""]
        drugbank = ["DB14577", "", ""]

        for n in range(len(self.ligands)):
            map = LigandMap.worker(self.ligands[n], self.chembl_map, self.chebi_map, self.drugbank_map)
            self.assertEqual(map.chembl_id, chembl[n])
            self.assertEqual(map.chebi_id, chebi[n])
            self.assertEqual(map.drugbank_id, drugbank[n])