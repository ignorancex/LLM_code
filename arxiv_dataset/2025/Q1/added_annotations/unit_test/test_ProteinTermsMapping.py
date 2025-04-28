import unittest
import resources.ProteinTermsMapping
import mock
from models import Protein, Model

class TestProteinTermsMapping(unittest.TestCase):
    """
    UnitTest for GO, InterPro, Pfam, CATH, SCOP, SCOP2, SCOP2B and pdbekb
    """
    def setUp(self):
        super(TestProteinTermsMapping, self).setUp()
        self.sifts_prefix = "/sifts_path"

        self.is_interpro = True
        self.is_go = True
        self.is_pfam = True
        self.is_cath = True
        self.is_scop = True
        self.is_scop2 = True
        self.is_scop2B = True
        self.is_pdbekb = True
        model = Model("EMD-0001", '6GH5')
        self.proteins = [
            Protein("EMD-0001", "1", sample_name="DNA-directed RNA polymerase subunit alpha", sample_organism="83333", pdb=[model], uniprot_id="P0A7Z4",
                    sequence="MQGSVTEFLKPRLVDIEQVSSTHAKVTLEPLERGFGHTLGNALRRILLSSMPGCAVTEVEIDGVLHEYSTKEGVQEDILEILLNLKG"
                             "LAVRVQGKDEVILTLNKSGIGPVTAADITHDGDVEIVKPQHVICHLTDENASISMRIKVQRGRGYVPASTRIHSEEDERPIGRLLVDA"
                             "CYSPVERIAYNVEAARVEQRTDLDKLVIEMETNGTIDPEEAIRRAATILAEQLEAFVDLRDVRQPEVKEEKPEFDPILLRPVDDLELT"
                             "VRSANCLKAEAIHYIGDLVQRTEVELLKTPNLGKKSLTEIKDVLASRGLSLGMRLENWPPASIADE")]

    @mock.patch('gzip.open')
    def test_execute(self, mock_gzip):
        mock.mock_open(mock_gzip, """\
        <entry xmlns="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd">
            <segment segId="6gh5_A_1_329" start="1" end="329">
                <listResidue>
                    <residue dbSource="PDBe" dbCoordSys="PDBe" dbResNum="58" dbResName="VAL">
                        <crossRefDb dbSource="PDB" dbCoordSys="PDBresnum" dbAccessionId="6gh5" dbResNum="58" dbResName="GLU" dbChainId="B"/>
                        <crossRefDb dbSource="UniProt" dbCoordSys="UniProt" dbAccessionId="P0A7Z4" dbResNum="58" dbResName="E"/>
                        <crossRefDb dbSource="Pfam" dbCoordSys="UniProt" dbAccessionId="PF01000" dbResNum="58" dbResName="E"/>
                        <crossRefDb dbSource="CATH" dbCoordSys="PDBresnum" dbAccessionId="2.170.120.12" dbResNum="58" dbResName="GLU" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP" dbCoordSys="PDBresnum" dbAccessionId="SF:2000112" dbResNum="58" dbResName="GLU" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP2" dbCoordSys="PDBresnum" dbAccessionId="SF:3000133" dbResNum="58" dbResName="GLU" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP2B" dbCoordSys="PDBresnum" dbAccessionId="SF-DOMID:8037279" dbResNum="58" dbResName="GLU" dbChainId="B"/>
                        <crossRefDb dbSource="InterPro" dbCoordSys="PDBresnum" dbAccessionId="IPR010243" dbResNum="58" dbResName="PRO" dbChainId="C" dbEvidence="TIGR02013"/>
                    </residue>
                    <residue dbSource="PDBe" dbCoordSys="PDBe" dbResNum="186" dbResName="ASN">
                        <crossRefDb dbSource="PDB" dbCoordSys="PDBresnum" dbAccessionId="6gh5" dbResNum="186" dbResName="ASN" dbChainId="B"/>
                        <crossRefDb dbSource="UniProt" dbCoordSys="UniProt" dbAccessionId="P0A7Z4" dbResNum="186" dbResName="N"/>
                        <crossRefDb dbSource="Pfam" dbCoordSys="UniProt" dbAccessionId="PF01000" dbResNum="186" dbResName="N"/>
                        <crossRefDb dbSource="CATH" dbCoordSys="PDBresnum" dbAccessionId="3.30.1360.10" dbResNum="186" dbResName="ASN" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP" dbCoordSys="PDBresnum" dbAccessionId="SF:2000112" dbResNum="186" dbResName="ASN" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP2" dbCoordSys="PDBresnum" dbAccessionId="SF:3000133" dbResNum="186" dbResName="ASN" dbChainId="B"/>
                        <crossRefDb dbSource="SCOP2B" dbCoordSys="PDBresnum" dbAccessionId="SF-DOMID:8035907" dbResNum="186" dbResName="ASN" dbChainId="B"/>
                        <crossRefDb dbSource="InterPro" dbCoordSys="PDBresnum" dbAccessionId="IPR010243" dbResNum="186" dbResName="PHE" dbChainId="C" dbEvidence="TIGR02013"/>
                    </residue>
                </listResidue>
                <listMapRegion>
                    <mapRegion start="1" end="329">
                        <db dbSource="UniProt" dbCoordSys="UniProt" dbAccessionId="P0A7Z4" start="1" end="329"/>
                    </mapRegion>
                    <mapRegion start="1" end="329">
                        <db dbSource="GO" dbCoordSys="UniProt" dbAccessionId="GO:0016020" dbEvidence="HDA"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="InterPro" dbCoordSys="PDBresnum" dbAccessionId="IPR011263"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="Pfam" dbCoordSys="UniProt" dbAccessionId="PF01000" coverage="1"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="CATH" dbCoordSys="PDBresnum" dbAccessionId="2.170.120.12"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="SCOP" dbCoordSys="PDBresnum" dbAccessionId="SF:2000112"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="SCOP2" dbCoordSys="PDBresnum" dbAccessionId="SF:3000133"/>
                    </mapRegion>
                    <mapRegion start="58" end="186">
                      <db dbSource="SCOP2B" dbCoordSys="PDBresnum" dbAccessionId="SF-DOMID:8035907"/>
                    </mapRegion>
                </listMapRegion>
            </segment>
        </entry>""")

        uniprot_with_models = {'A0A0J9X294', 'P0A7Z4', 'P0A800', 'R4GRT5'}
        ProteinMap = resources.ProteinTermsMapping.ProteinTermsMapping(self.proteins, self.sifts_prefix, self.is_go, self.is_interpro,
                                                                       self.is_pfam, self.is_cath, self.is_scop,
                                                                       self.is_scop2, self.is_scop2B, self.is_pdbekb)
        css = [('P0A7Z4', 'PDBe', '58', '186', '58', '186')]
        go = [('GO:0016020', 'membrane', 'C', 'P0A7Z4', 'PDBe')]
        interpro = [('IPR011263', 'DNA-dir_RNA_pol_RpoA/D/Rpb3', 'P0A7Z4', 'PDBe', '58', '186', '58', '186')]
        pfam = [('PF01000','RNA_pol_A_bac', 'P0A7Z4', 'PDBe', '58', '186', '58', '186')]

        pdbekb = [('P0A7Z4', 'UniProt'), ('P0A800', 'UniProt')]
        for n in range(len(self.proteins)):
            PT = ProteinMap.execute(uniprot_with_models)
            for i in PT[n].go:
                self.assertEqual(i.id, go[n][0])
                self.assertEqual(i.namespace, go[n][1])
                self.assertEqual(i.type, go[n][2])
                self.assertEqual(i.unip_id, go[n][3])
                self.assertEqual(i.provenance, go[n][4])
            for i in PT[n].interpro:
                self.assertEqual(i.id, interpro[n][0])
                self.assertEqual(i.namespace, interpro[n][1])
                self.assertEqual(i.unip_id, interpro[n][2])
                self.assertEqual(i.provenance, interpro[n][3])
                self.assertEqual(i.start, int(interpro[n][4]))
                self.assertEqual(i.end, int(interpro[n][5]))
                self.assertEqual(i.unp_start, int(interpro[n][6]))
                self.assertEqual(i.unp_end, int(interpro[n][7]))
            for i in PT[n].pfam:
                self.assertEqual(i.id, pfam[n][0])
                self.assertEqual(i.namespace, pfam[n][1])
                self.assertEqual(i.unip_id, pfam[n][2])
                self.assertEqual(i.provenance, pfam[n][3])
                self.assertEqual(i.start, int(pfam[n][4]))
                self.assertEqual(i.end, int(pfam[n][5]))
                self.assertEqual(i.unp_start, int(pfam[n][6]))
                self.assertEqual(i.unp_end, int(pfam[n][7]))
            for i in PT[n].cath:
                self.assertEqual(i.id, "2.170.120.12")
                self.assertEqual(i.unip_id, css[n][0])
                self.assertEqual(i.provenance, css[n][1])
                self.assertEqual(i.start, int(css[n][2]))
                self.assertEqual(i.end, int(css[n][3]))
                self.assertEqual(i.unp_start, int(css[n][4]))
                self.assertEqual(i.unp_end, int(css[n][5]))
            for i in PT[n].scop:
                self.assertEqual(i.id, "SF:2000112")
                self.assertEqual(i.unip_id, css[n][0])
                self.assertEqual(i.provenance, css[n][1])
                self.assertEqual(i.start, int(css[n][2]))
                self.assertEqual(i.end, int(css[n][3]))
                self.assertEqual(i.unp_start, int(css[n][4]))
                self.assertEqual(i.unp_end, int(css[n][5]))
            for i in PT[n].scop2:
                self.assertEqual(i.id, "SF:3000133")
                self.assertEqual(i.unip_id, css[n][0])
                self.assertEqual(i.provenance, css[n][1])
                self.assertEqual(i.start, int(css[n][2]))
                self.assertEqual(i.end, int(css[n][3]))
                self.assertEqual(i.unp_start, int(css[n][4]))
                self.assertEqual(i.unp_end, int(css[n][5]))
            for i in PT[n].scop2B:
                self.assertEqual(i.id, "SF-DOMID:8035907")
                self.assertEqual(i.unip_id, css[n][0])
                self.assertEqual(i.provenance, css[n][1])
                self.assertEqual(i.start, int(css[n][2]))
                self.assertEqual(i.end, int(css[n][3]))
                self.assertEqual(i.unp_start, int(css[n][4]))
                self.assertEqual(i.unp_end, int(css[n][5]))
            self.assertEqual(PT[n].pdbekb.unip_id, pdbekb[n][0])
            self.assertEqual(PT[n].pdbekb.provenance, pdbekb[n][1])