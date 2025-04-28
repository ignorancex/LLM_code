import re
import requests
import json

class Protein:
    """
    Defines the attributes of a protein sample in a EMDB entry
    """
    def __init__(self, emdb_id, sample_id, sample_name="", sample_organism=None, pdb=None, sample_complexes=None, uniprot_id=None,
                 provenance=None, sequence="", sample_copies="", go=None, interpro=None, pfam=None, cath=None, scop=None,
                 scop2=None, scop2B=None, pdbekb=None, alphafold=None):
        self.emdb_id = emdb_id
        self.sample_id = sample_id
        self.sample_name = sample_name
        self.sample_organism = sample_organism
        self.pdb = [] if pdb is None else pdb
        self.sample_complexes = [] if sample_complexes is None else sample_complexes
        self.uniprot_id = uniprot_id
        self.provenance = provenance
        self.sequence = sequence
        self.sample_copies = sample_copies
        self.go = set() if go is None else go
        self.interpro = set() if interpro is None else interpro
        self.pfam = set() if pfam is None else pfam
        self.cath = set() if cath is None else cath
        self.scop = set() if scop is None else scop
        self.scop2 = set() if scop2 is None else scop2
        self.scop2B = set() if scop2B is None else scop2B
        self.pdbekb = pdbekb
        self.alphafold = alphafold

    def __str__(self):
        return "%s (%s)\n%s (%s) %s - %s [%s]\nComplexes: %s\nPDB: \n%s\n%s\n%s\n  %s\n%s\n" % (self.sample_name, self.sample_organism,
                                                                              self.emdb_id, self.sample_id, self.sample_copies,
                                                                              self.uniprot_id, self.provenance, str(self.sample_complexes),
                                                                              str(self.go), str(self.interpro), str(self.pfam),
                                                                              str(self.pdbekb), str(self.alphafold))

    def get_tsv(self):
        complex_str = ';'.join([str(elem) for elem in self.sample_complexes])
        if self.provenance:
            return ("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.emdb_id, self.sample_id, self.sample_name, self.sample_copies,
                                                          self.sample_organism, self.uniprot_id, self.provenance, complex_str))
        else:
            return ""

class CPX:
    """
    Complex Portal entry obtained from their FTP area
    """

    def __init__(self, row):
        self.cpx_id = row[0]
        self.name = row[1]
        self.taxonomy = row[3]
        self.uniprot = set()
        self.identifiers = re.sub(r'\(\d+\)', '', row[4]).split('|')
        self.confidence = row[5]
        self.GO = re.sub(r'\(.+?\)', '', row[7]).split('|')
        self.cross_ref = re.findall(r':(.*?)\(', row[8], re.S)

        for idt in self.identifiers:
            if 'CHEBI:' in idt:
                continue
            if '-PRO_' in idt:
                self.uniprot.add(idt.split('-')[0])
                continue
            if  '_' in idt:
                continue
            self.uniprot.add(idt)

class Sample:
    """
    Unique sample along its parents and childrens
    """
    def __init__(self, sample_id, mw=None, copies=1):
        self.id = sample_id
        self.mw = mw
        self.parent = []
        self.children = []
        self.copies = copies

    def add_parent(self, node):
        self.parent.append(node)

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        return f"{self.id}: {self.mw} ({self.copies})"

class Supramolecule:
    """
    Defines the attributes of a supra_molecules in a EMDB entry
    """
    def __init__(self, emdb_id, supramolecule_id, name="", mol_type=""):
        self.id = supramolecule_id
        self.emdb_id = emdb_id
        self.name = name
        self.type = mol_type

    def __str__(self):
        return "%s\t%s\t%s\t%s" % (self.emdb_id, self.id, self.name, self.type)

class EMDB_complex:
    """
    EMDB complex sample obtained from the header files in the Uniprot mapping
    """

    def __init__(self, emdb_id, sample_id, name, sample_copies, complex_sample_id, cpx_list=None, proteins=None,
                 provenance="", score=0.0):
        self.emdb_id = emdb_id
        self.sample_id = emdb_id+"_"+sample_id
        self.name = name
        self.sample_copies = sample_copies
        self.complex_sample_id = complex_sample_id
        self.cpx_list = [] if cpx_list is None else cpx_list
        self.proteins = set() if proteins is None else proteins
        self.provenance = provenance
        self.score = score

    def add_protein(self, uniprot_id):
        self.proteins.add(uniprot_id)

class Ligand:
    """
    Defines the attributes of a ligands sample in a EMDB entry
    """
    def __init__(self, emdb_id, sample_id, chembl_id="", chebi_id="", drugbank_id="", provenance_chembl="", provenance_chebi="",
                 provenance_drugbank="", HET="", name="", copies=1):
        self.emdb_id = emdb_id
        self.sample_id = sample_id
        self.provenance_chebi = provenance_chebi
        self.provenance_chembl = provenance_chembl
        self.provenance_drugbank = provenance_drugbank
        self.HET = HET
        self.name = name
        self.chembl_id = chembl_id
        self.chebi_id = chebi_id
        self.drugbank_id = drugbank_id
        self.copies = copies

    def get_chembl_tsv(self):
        if self.chembl_id:
            return "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.emdb_id, self.sample_id, self.HET, self.name,
                                                     self.copies, self.chembl_id, self.provenance_chembl)
        return ""

    def get_chebi_tsv(self):
        if self.chebi_id:
            return "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.emdb_id, self.sample_id, self.HET, self.name,
                                                     self.copies, self.chebi_id, self.provenance_chebi)
        return ""

    def get_drugbank_tsv(self):
        if self.drugbank_id:
            return "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.emdb_id, self.sample_id, self.HET, self.name,
                                                     self.copies, self.drugbank_id, self.provenance_drugbank)
        return ""

class Model:
    """
    Define the PDB model, preferred assembly and molecular weight
    """
    def __init__(self, emdb_id, pdb_id, assembly=1, molecular_weight=0.0):
        self.emdb_id = emdb_id
        self.pdb_id = pdb_id
        self.assembly = assembly
        self.molecular_weight = molecular_weight #Dalton

    def __str__(self):
        return ("{}\t{}\t{}\t{:0.3f}".format(self.emdb_id, self.pdb_id, self.assembly, self.molecular_weight))

class Weight:
    """
    Total weight of the sample provided by author
    """
    def __init__(self, emdb_id, overall_mw=0.0, units="", provenance=""):
        self.emdb_id = emdb_id
        self.overall_mw = overall_mw
        self.units = units
        self.provenance = provenance

    def __str__(self):
            return ("%s\t%s\t%s\t%s\n" % (self.emdb_id, self.overall_mw, self.units, self.provenance))

class Empiar:
    """
    Defines the EMPIAR ID in a EMDB entry
    """
    def __init__(self, emdb_id, empiar_id=""):
        self.emdb_id = emdb_id
        self.empiar_id = empiar_id

    def __str__(self):
        return "%s\t%s\n" % (self.emdb_id, self.empiar_id)

class Author:
    """
    Defines the attributes of an author
    """
    def __init__(self, name, order, orcid="", provenance="EMDB"):
        self.name = name
        self.order = order
        self.orcid = orcid
        self.provenance = provenance
    def __str__(self):
        return f"{self.name}\t{self.orcid}\t{self.order}\t{self.provenance}"

class Citation:
    """
    Defines the attributes of a publication in a EMDB entry
    """
    def __init__(self, emdb_id, pmedid="", pmcid="", doi="", issn="", journal="", journal_abbv="", authors=None, published=True,
                 title="", provenance_pm="", provenance_pmc="", provenance_issn="", provenance_doi="", provenance_orcid="",
                 url=""):
        self.emdb_id = emdb_id
        self.pmedid = pmedid
        self.pmcid = pmcid
        self.doi = doi
        self.issn = issn
        self.journal = journal
        self.journal_abbv = journal_abbv
        self.authors = [] if authors is None else authors
        self.published = published
        self.title = title
        self.provenance_pm = provenance_pm
        self.provenance_pmc = provenance_pmc
        self.provenance_issn = provenance_issn
        self.provenance_doi = provenance_doi
        self.provenance_orcid = provenance_orcid
        self.url = url

    def __str__(self):
        return f"{self.emdb_id}\t{self.pmedid}\t{self.provenance_pm}\t{self.pmcid}\t{self.provenance_pmc}\t{self.issn}\t{self.provenance_issn}\t{self.doi}\t{self.provenance_doi}\t{self.journal}\t{self.journal_abbv}"
    def addExternalOrcid(self, orcid, order, provenance):
        for author in self.authors:
            if author.order == order:
                author.orcid = orcid
                author.provenance = provenance

class GO:
    """
    Define the GO terms for the sample in the EMDB entry
    """
    def __init__(self, id="", namespace="", type="", unip_id="", provenance=""):
        self.id = id
        self.namespace = namespace
        self.type = type
        self.unip_id = unip_id
        self.provenance = provenance

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id)

    def __hash__(self):
        return hash(self.id + self.namespace + self.unip_id + self.provenance + self.type)


    def __str__(self):
        return f"{self.id}\t{self.namespace}\t{self.type}\t{self.provenance}"

    def add_from_author(self, go_text, unip_id):
        self.provenance = "EMDB"
        self.unip_id = unip_id
        if "GO:" in go_text:
            self.id = go_text
        elif go_text.isdigit():
            self.id = f"GO:{go_text}"

        if self.id and not self.namespace:
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{self.id}"
            try:
                response = requests.get(url, timeout=10)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
                return False
            if response.status_code == 200:
                res_text = response.text
                data = json.loads(res_text)
                hits = data['numberOfHits']
                if hits > 0:
                    result = data['results'][0]
                    self.namespace = result['name']
                    aspect = result['aspect']
                    if aspect == 'biological_process':
                        self.type = "P"
                    elif aspect == 'cellular_component':
                        self.type = "C"
                    elif aspect == 'molecular_function':
                        self.type = "F"


class Interpro:
    """
    Define the InterPro terms for the sample in the EMDB entry
    """
    def __init__(self, id="", namespace="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.namespace = namespace
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.namespace + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))

    def __str__(self):
        return f"{self.id}\t{self.namespace}\t{self.provenance}"

    def add_from_author(self, ipr_text, unip_id):
        self.provenance = "EMDB"
        self.unip_id = unip_id
        if "IPR" in ipr_text:
            self.id = ipr_text

        if self.id and not self.namespace:
            url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{self.id}"
            try:
                response = requests.get(url, timeout=10)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
                return False
            if response.status_code == 200:
                res_text = response.text
                data = json.loads(res_text)
                if 'metadata' in data:
                    result = data['metadata']
                    if 'hierarchy' in result:
                        if 'name' in result['hierarchy']:
                            self.namespace = result['hierarchy']['name']

class Pfam:
    """
    Define the Pfam domains for the sample in the EMDB entry
    """
    def __init__(self, id="", namespace="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.namespace = namespace
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.namespace + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))


    def __str__(self):
        return f"{self.id}\t{self.namespace}\t{self.provenance}"

    def add_from_author(self, pfam_text, unip_id):
        self.provenance = "EMDB"
        self.unip_id = unip_id
        if "PF" in pfam_text:
            self.id = pfam_text

        if self.id and not self.namespace:
            url = f"https://pfam.xfam.org/family/{self.id}?output=xml"
            try:
                response = requests.get(url, timeout=10)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
                return False
            if response.status_code == 200:
                res_text = response.text
                data = json.loads(res_text)
                if 'description' in data:
                    result = data['description']
                    self.namespace = result['description']

class Cath:
    """
    Define the CATH domains for the sample in the EMDB entry
    """
    def __init__(self, id="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))

    def __str__(self):
        return f"{self.id}\t{self.start}\t{self.end}\t{self.provenance}"

class SCOP:
    """
    Define the SCOP domains for the sample in the EMDB entry
    """
    def __init__(self, id="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))

    def __str__(self):
        return f"{self.id}\t{self.start}\t{self.end}\t{self.provenance}"

class SCOP2:
    """
    Define the SCOP2 domains for the sample in the EMDB entry
    """
    def __init__(self, id="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))

    def __str__(self):
        return f"{self.id}\t{self.start}\t{self.end}\t{self.provenance}"

class SCOP2B:
    """
    Define the SCOP2B domains for the sample in the EMDB entry
    """
    def __init__(self, id="", unip_id="", provenance="", start=0, end=0, unp_start=0, unp_end=0):
        self.id = id
        self.unip_id = unip_id
        self.provenance = provenance
        self.start = start
        self.end = end
        self.unp_start = unp_start
        self.unp_end = unp_end

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'id', None) == self.id and
            getattr(other, 'start', None) == self.start and
            getattr(other, 'end', None) == self.end and
            getattr(other, 'unp_start', None) == self.unp_start and
            getattr(other, 'unp_end', None) == self.unp_end)

    def __hash__(self):
        return hash(self.id + self.unip_id + self.provenance + str(self.start) + str(self.end) + str(self.unp_start) + str(self.unp_end))

    def __str__(self):
        return f"{self.id}\t{self.start}\t{self.end}\t{self.provenance}"


class Pdbekb:
    """
    Define the PDBeKB terms for the sample in the EMDB entry
    """
    def __init__(self, uniprot_id, provenance=None):
        self.unip_id = uniprot_id
        self.provenance = provenance

    def __str__(self):
        return f"{self.unip_id}\t{self.provenance}"

class Alphafold:
    """
    Define the Alphafold terms for the sample in the EMDB entry
    """
    def __init__(self, uniprot_id, provenance=None):
        self.unip_id = uniprot_id
        self.provenance = provenance

    def __str__(self):
        return f"{self.unip_id}\t{self.provenance}"

class Rfam:
    """
    Define the Rfam accession and id for the sample id in the EMDB entry
    """
    def __init__(self, emdb_id, sample_id, sample_name="", num_copies=1, pdb_id=[], rfam_acc="", rfam_id="", provenance=""):
        self.emdb_id = emdb_id
        self.sample_id = sample_id
        self.sample_name = sample_name
        self.num_copies = num_copies
        self.pdb_id = pdb_id
        self.rfam_acc = rfam_acc
        self.rfam_id = rfam_id
        self.provenance = provenance

    def __str__(self):
        return f"{self.emdb_id}\t{self.sample_id}\t{self.sample_name}\t{self.num_copies}\t{self.rfam_acc}\t{self.rfam_id}\t{self.provenance}"
