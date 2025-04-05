from jarvis.db.figshare import get_jid_data
from chemnlp.utils.process_doc import ProcessDoc
from chemnlp.utils.describe import atoms_describer 
from jarvis.core.atoms import Atoms
import pprint

def test_formula():
    """Test formula."""
    txt = "MgB2 is a superconductor."
    pdoc = ProcessDoc(text=txt)
    print(pdoc.get_chemical_formulae())
def test_describer():
    atoms = Atoms.from_dict(
        get_jid_data(jid="JVASP-32", dataset="dft_3d")["atoms"]
    )
    info = atoms_describer(atoms=atoms)
    pprint.pprint(info)
