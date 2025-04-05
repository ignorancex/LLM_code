"""Module to parse documents."""

from jarvis.core.composition import Composition
from chemdataextractor import Document
import math


class ProcessDoc(object):
    """Module to parse documents."""

    def __init__(self, doc_id=None, text=""):
        """Initialize the class."""
        self.text = text
        self.doc_id = doc_id

    def get_chemical_formulae(self):
        """Get chemical formulae."""
        formulae = []
        doc = Document(self.text)
        for j in doc.cems:
            wt = "na"
            try:
                form = Composition.from_string(j.text)
                wt = float(form.weight)
            except Exception:
                pass
            if (
                wt != "na"
                and not math.isnan(wt)
                # and form.nspecies > 1
                # and "cond-mat" in i["categories"]
            ):
                # print(i['id'],i['title'],j,form.reduced_formula)

                if form.reduced_formula not in formulae:
                    formulae.append(form.reduced_formula)
        return formulae


if __name__ == "__main__":
    txt = "MgB2 is a superconductor."
    pdoc = ProcessDoc(text=txt)
    print(pdoc.get_chemical_formulae())
