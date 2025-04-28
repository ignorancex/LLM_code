from EMICSS.EMICSS import *
from models import *
import os
from pathlib import Path

class Parser:
    """
    Read log files and create emicss objects
    """
    def __init__(self, workDir):
        self.workDir = workDir
        self.emdb_ids = set()
        self.proteins = {}
        self.empiars = {}
        self.weights = {}
        self.models = {}
        self.citations = {}
        self.ligands = {}
        self.complexes = {}
        self.rna = {}
        self.__parse_uniprot()
        self.__parse_empiar()
        self.__parse_mw()
        self.__parse_models()
        self.__parse_citations()
        self.__parse_chembl()
        self.__parse_chebi()
        self.__parse_drugbank()
        self.__parse_complex()
        self.__parse_go()
        self.__parse_interpro()
        self.__parse_pfam()
        self.__parse_cath()
        self.__parse_scop()
        self.__parse_scop2()
        self.__parse_scop2B()
        self.__parse_pdbekb()
        self.__parse_afdb()
        self.__parse_rfam()

    def get_packed_data(self, emdb_id):
        return {
            'emdb_id': emdb_id,
            'proteins': self.proteins.get(emdb_id), # {sample_id: Protein}
            'empiar': self.empiars.get(emdb_id), # [Empiar]
            'weight': self.weights.get(emdb_id), # Weight
            'models': self.models.get(emdb_id), # [Model]
            'citation': self.citations.get(emdb_id), # Citation
            'ligands': self.ligands.get(emdb_id), # {sample_id: Ligand}
            'complexes': self.complexes.get(emdb_id), # {sample_id: EMDB_Complex}
            'rna': self.rna.get(emdb_id) # {sample_id: RNA}
        }

    def __parse_rfam(self):
        fileafdb = os.path.join(self.workDir, 'emdb_rfam.log')
        with open(fileafdb, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, sample_name, num_copies, rfam_acc, rfam_id, provenance = line.strip('\n').split('\t')
                rna = Rfam(emdb_id=emdb_id, sample_id=sample_id, sample_name=sample_name, num_copies=num_copies, rfam_acc=rfam_acc,
                            rfam_id=rfam_id, provenance=provenance)
                self.__add_rna(emdb_id, sample_id, sample_name, num_copies, rna, rfam_acc, rfam_id, provenance)

    def __parse_uniprot(self):
        fileuniprot = os.path.join(self.workDir, 'emdb_uniprot.log')
        with open(fileuniprot, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, sample_name, copies, ncbi_id, uniprot_id, provenance, sample_complex_ids = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                protein = Protein(emdb_id=emdb_id, sample_id=sample_id, sample_name=sample_name, sample_copies=copies, sample_organism=ncbi_id, uniprot_id=uniprot_id, provenance=provenance)
                if emdb_id in self.proteins:
                    self.proteins[emdb_id][sample_id] = protein
                else:
                    self.proteins[emdb_id] = {sample_id: protein}

    def __parse_empiar(self):
        fileempiar = os.path.join(self.workDir, "emdb_empiar.log")
        with open(fileempiar, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, empiar_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                empiar = Empiar(emdb_id=emdb_id, empiar_id=empiar_id)
                if emdb_id in self.empiars:
                    self.empiars[emdb_id].append(empiar)
                else:
                    self.empiars[emdb_id] = [empiar]

    def __parse_mw(self):
        filemw = os.path.join(self.workDir, "overall_mw.log")
        with open(filemw, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, mw = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                weight = Weight(emdb_id=emdb_id, overall_mw=float(mw), units="MDa", provenance="EMDB")
                self.weights[emdb_id] = weight

    def __parse_models(self):
        filemodel = os.path.join(self.workDir, "emdb_model.log")
        with open(filemodel, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, pdb_id, assembly, mw = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                model = Model(emdb_id=emdb_id, pdb_id=pdb_id, assembly=assembly, molecular_weight=float(mw))
                if emdb_id in self.models:
                    self.models[emdb_id].append(model)
                else:
                    self.models[emdb_id] = [model]

    def __parse_citations(self):
        author_dict = {}
        fileauthor = os.path.join(self.workDir, 'emdb_author.log')
        with open(fileauthor, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, author_name, orcid_id, order, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                author = Author(name=author_name, orcid=orcid_id, order=order, provenance=provenance)
                if emdb_id in author_dict:
                    author_dict[emdb_id].append(author)
                else:
                    author_dict[emdb_id] = [author]
        filepubmed = os.path.join(self.workDir, "emdb_pubmed.log")
        with open(filepubmed, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, pubmed_id, pubmed_provenance, pmc_id, pmc_provenance, issn, issn_provenance, doi, doi_provenance, journal_name, journal_abbv = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                authors_list = author_dict[emdb_id]
                citation = Citation(emdb_id=emdb_id, pmedid=pubmed_id, provenance_pm=pubmed_provenance, pmcid=pmc_id, 
                                    provenance_pmc=pmc_provenance, issn=issn, provenance_issn=issn_provenance, doi=doi, 
                                    provenance_doi=doi_provenance, authors=authors_list)
                self.citations[emdb_id] = citation

    def __parse_chembl(self):
        filechembl = os.path.join(self.workDir, 'emdb_chembl.log')
        with open(filechembl, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, het_code, ligand_name, copies, chembl_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                ligand = Ligand(emdb_id=emdb_id, sample_id=sample_id, chembl_id=chembl_id,
                                provenance_chembl=provenance, HET=het_code, name=ligand_name,
                                copies=copies)
                self.__add_ligand(emdb_id, sample_id, ligand, chembl_id, provenance, "ChEMBL")

    def __parse_chebi(self):
        filechebi = os.path.join(self.workDir, 'emdb_chebi.log')
        with open(filechebi, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, het_code, ligand_name, copies, chebi_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                ligand = Ligand(emdb_id=emdb_id, sample_id=sample_id, chebi_id=chebi_id,
                                provenance_chebi=provenance, HET=het_code, name=ligand_name, copies=copies)
                self.__add_ligand(emdb_id, sample_id, ligand, chebi_id, provenance, "ChEBI")

    def __parse_drugbank(self):
        filedb = os.path.join(self.workDir, 'emdb_drugbank.log')
        with open(filedb, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, het_code, ligand_name, copies, db_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                ligand = Ligand(emdb_id=emdb_id, sample_id=sample_id, drugbank_id=db_id,
                                provenance_drugbank=provenance, HET=het_code, name=ligand_name, copies=copies)
                self.__add_ligand(emdb_id, sample_id, ligand, db_id, provenance, "DrugBank")

    def __parse_complex(self):
        filecpx = os.path.join(self.workDir, "emdb_cpx.log")
        with open(filecpx, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, sample_name, copies, cpx_id, cpx_title, provenance, score = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                cpx = CPX([cpx_id, cpx_title, "", "", "", "", "", "", ""])
                emdb_complex = EMDB_complex(emdb_id=emdb_id, sample_id=sample_id, name=sample_name,
                             sample_copies=copies, complex_sample_id=sample_id, cpx_list=[cpx],
                             proteins=None, provenance=provenance, score=float(score))
                self.__add_complex(emdb_id, sample_id, emdb_complex, cpx)

    def __parse_go(self):
        filego = os.path.join(self.workDir, 'emdb_go.log')
        with open(filego, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, go_id, go_namespace, go_type, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                go = GO(id=go_id, namespace=go_namespace, type=go_type, provenance=provenance)
                self.proteins[emdb_id][sample_id].go.add(go)

    def __parse_interpro(self):
        fileinterpro = os.path.join(self.workDir, 'emdb_interpro.log')
        with open(fileinterpro, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, interpro_id, interpro_namespace, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                interpro = Interpro(id=interpro_id, namespace=interpro_namespace, start=int(start), end=int(end),
                                    unp_start=int(uniprot_start), unp_end=int(uniprot_end), provenance=provenance)
                self.proteins[emdb_id][sample_id].interpro.add(interpro)

    def __parse_pfam(self):
        filepfam = os.path.join(self.workDir, 'emdb_pfam.log')
        with open(filepfam, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, pfam_id, pfam_namespace, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                pfam = Pfam(id=pfam_id, namespace=pfam_namespace, start=int(start), end=int(end), unp_start=int(uniprot_start),
                            unp_end=int(uniprot_end), provenance=provenance)
                self.proteins[emdb_id][sample_id].pfam.add(pfam)

    def __parse_cath(self):
        filecath = os.path.join(self.workDir, 'emdb_cath.log')
        with open(filecath, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, cath_id, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                cath = Cath(id=cath_id, start=int(start), end=int(end), unp_start=int(uniprot_start), unp_end=int(uniprot_end),
                            provenance=provenance)
                self.proteins[emdb_id][sample_id].cath.add(cath)

    def __parse_scop(self):
        filescop = os.path.join(self.workDir, 'emdb_scop.log')
        with open(filescop, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, scop_id, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                scop = SCOP(id=scop_id, start=int(start), end=int(end), unp_start=int(uniprot_start), unp_end=int(uniprot_end),
                            provenance=provenance)
                self.proteins[emdb_id][sample_id].scop.add(scop)

    def __parse_scop2(self):
        filescop = os.path.join(self.workDir, 'emdb_scop2.log')
        with open(filescop, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, scop_id, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                scop2 = SCOP2(id=scop_id, start=int(start), end=int(end), unp_start=int(uniprot_start), unp_end=int(uniprot_end),
                            provenance=provenance)
                self.proteins[emdb_id][sample_id].scop2.add(scop2)

    def __parse_scop2B(self):
        filescop = os.path.join(self.workDir, 'emdb_scop2B.log')
        with open(filescop, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, scop_id, start, end, uniprot_start, uniprot_end, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                scop2b = SCOP2B(id=scop_id, start=int(start), end=int(end), unp_start=int(uniprot_start), unp_end=int(uniprot_end),
                            provenance=provenance)
                self.proteins[emdb_id][sample_id].scop2B.add(scop2b)

    def __parse_pdbekb(self):
        filepdbekb = os.path.join(self.workDir, 'emdb_pdbekb.log')
        with open(filepdbekb, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, pdbekb_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                pdbekb = Pdbekb(uniprot_id=pdbekb_id, provenance=provenance)
                self.proteins[emdb_id][sample_id].pdbekb = pdbekb

    def __parse_afdb(self):
        fileafdb = os.path.join(self.workDir, 'emdb_alphafold.log')
        with open(fileafdb, 'r') as filereader:
            for line in filereader.readlines()[1:]:
                emdb_id, sample_id, afdb_id, provenance = line.strip('\n').split('\t')
                self.emdb_ids.add(emdb_id)
                afdb = Alphafold(uniprot_id=afdb_id, provenance=provenance)
                self.proteins[emdb_id][sample_id].alphafold = afdb

    def __add_ligand(self, emdb_id, sample_id, ligand, ligand_id, provenance, ligand_type):
        if emdb_id in self.ligands:
            if sample_id in self.ligands:
                if ligand_type == "ChEMBL":
                    self.ligands[emdb_id][sample_id].chembl_id = ligand_id
                    self.ligands[emdb_id][sample_id].provenance_chembl = provenance
                elif ligand_type == "ChEBI":
                    self.ligands[emdb_id][sample_id].chebi_id = ligand_id
                    self.ligands[emdb_id][sample_id].provenance_chebi = provenance
                elif ligand_type == "DrugBank":
                    self.ligands[emdb_id][sample_id].drugbank_id = ligand_id
                    self.ligands[emdb_id][sample_id].provenance_drugbank = provenance
            else:
                self.ligands[emdb_id][sample_id] = ligand
        else:
            self.ligands[emdb_id] = {sample_id: ligand}

    def __add_complex(self, emdb_id, sample_id, emdb_complex, cpx):
        if emdb_id in self.complexes:
            if sample_id in self.complexes[emdb_id]:
                self.complexes[emdb_id][sample_id].cpx_list.append(cpx)
            else:
                self.complexes[emdb_id][sample_id] = emdb_complex
        else:
            self.complexes[emdb_id] = {sample_id: emdb_complex}

    def __add_rna(self, emdb_id, sample_id, sample_name, num_copies, rna, rfam_acc, rfam_id, provenance):
        if emdb_id in self.rna:
            if sample_id in self.rna[emdb_id]:
                self.rna[emdb_id][sample_id].rfam_acc = rfam_acc
                self.rna[emdb_id][sample_id].rfam_id = rfam_id
                self.rna[emdb_id][sample_id].sample_name = sample_name
                self.rna[emdb_id][sample_id].num_copies = num_copies
                self.rna[emdb_id][sample_id].provenance = provenance
            else:
                self.rna[emdb_id][sample_id] = rna
        else:
            self.rna[emdb_id] = {sample_id: rna}

class EmicssXML:
    """
    Store and Write single EMICSS data
    """
    def __init__(self, emdb_id, version, packed_data):
        self.emdb_id = emdb_id
        self.versions = version
        self.headerXML = emicss(emdb_id=emdb_id)
        self.dbs = dbsType(collection_date=version.today)
        self.entry_ref_dbs = entry_ref_dbsType()
        self.weights = weightsType()
        self.sample = sampleType()
        self.macromolecules = macromoleculesType()
        self.supramolecules = supramoleculesType()
        self.primary_citation = primary_citationType()
        self.all_db = set()

        self.__read_empiar(packed_data.get("empiar"))
        self.__read_weight(packed_data.get("weight"))
        self.__read_models(packed_data.get("models"))
        self.__read_citation(packed_data.get("citation"))
        self.__read_proteins(packed_data.get("proteins"))
        self.__read_ligands(packed_data.get("ligands"))
        self.__read_complexes(packed_data.get("complexes"))
        self.__read_rna(packed_data.get("rna"))

        self.__create_dbs()

        if self.all_db:
            self.headerXML.set_dbs(self.dbs)
        if self.entry_ref_dbs.hasContent_():
            self.headerXML.set_entry_ref_dbs(self.entry_ref_dbs)
        if self.primary_citation.hasContent_():
            self.headerXML.set_primary_citation(self.primary_citation)
        if self.weights.hasContent_():
            self.headerXML.set_weights(self.weights)
        if self.supramolecules.hasContent_():
            self.sample.set_supramolecules(self.supramolecules)
        if self.macromolecules.hasContent_():
            self.sample.set_macromolecules(self.macromolecules)
        if self.sample.hasContent_():
            self.headerXML.set_sample(self.sample)

    def write_xml(self, work_dir):
        output_path = os.path.join(work_dir, "emicss")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        xmlFile = os.path.join(output_path, f"emd_{self.emdb_id[4:]}_emicss.xml")
        with open(xmlFile, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            self.headerXML.export(f, 0, name_='emicss', namespacedef_=f'version="{self.headerXML.version}" '
                                                                 f'schema_location="https://github.com/emdb-empiar/emicss-schema/tree/main/versions/emdb_emicss_{self.headerXML.version}.xsd" ')

    def __read_empiar(self, empiar_list):
        if empiar_list:
            self.all_db.add("EMPIAR")
            for empiar in empiar_list:
                emp_ref_db_obj = entry_ref_dbType(source="EMPIAR", accession_id=empiar.empiar_id, provenance="EMDB")
                self.entry_ref_dbs.add_entry_ref_db(emp_ref_db_obj)

    def __read_weight(self, mw):
        if mw:
            self.all_db.add("EMDB")
            mw_info_obj = weight_infoType(weight=round(mw.overall_mw, 2), unit=mw.units, provenance=mw.provenance)
            self.weights.add_weight_info(mw_info_obj)

    def __read_models(self, models):
        if models:
            self.all_db.add("EMDB")
            for pdb in models:
                pdb_info_obj = weight_infoType(pdb_id=pdb.pdb_id, assemblies=pdb.assembly,
                                               weight=pdb.molecular_weight, unit="Da", provenance="PDBe")
                self.weights.add_weight_info(pdb_info_obj)

    def __read_citation(self, citation):
        if citation:
            authors_obj = authorsType()
            for author in citation.authors:
                orcid = author.orcid if author.orcid else None
                author_obj = authorType(name=author.name, orcid_id=orcid, order=author.order,
                                        provenance=author.provenance)
                authors_obj.add_author(author_obj)
            self.primary_citation.set_authors(authors_obj)
            if citation.pmedid:
                self.all_db.add("PubMed")
                pm_citation_obj = ref_citationType(source="PubMed", accession_id=citation.pmedid,
                                                   provenance=citation.provenance_pm)
                self.primary_citation.add_ref_citation(pm_citation_obj)
                if citation.pmcid:
                    self.all_db.add("PubMed Central")
                    pmc_citation_obj = ref_citationType(source="PubMed Central", accession_id=citation.pmcid,
                                                        provenance=citation.provenance_pmc)
                    self.primary_citation.add_ref_citation(pmc_citation_obj)
                if citation.issn:
                    self.all_db.add("ISSN")
                    if citation.provenance_issn == "":
                        citation.provenance_issn = "EMDB"
                    issn_citation_obj = ref_citationType(source="ISSN", accession_id=citation.issn,
                                                         provenance=citation.provenance_issn)
                    self.primary_citation.add_ref_citation(issn_citation_obj)
                if citation.doi:
                    self.primary_citation.set_doi(citation.doi)
                    self.primary_citation.set_provenance(citation.provenance_doi)

    def __read_proteins(self, proteins):
        if proteins:
            self.all_db.add("EMDB")
            for sample_id, protein in proteins.items():
                cross_ref_dbs = cross_ref_dbsType()
                if protein.uniprot_id:
                    self.all_db.add("UniProt")
                    unp_xref_obj = cross_ref_db(source="UniProt", accession_id=protein.uniprot_id,
                                                provenance=protein.provenance)
                    cross_ref_dbs.add_cross_ref_db(unp_xref_obj)
                if protein.go:
                    self.all_db.add("GO")
                    for go in protein.go:
                        go_xref_obj = cross_ref_db(name=go.namespace, source="GO", accession_id=go.id,
                                                   provenance=go.provenance)
                        if go.type == "P":
                            go_xref_obj.set_type("biological process")
                        elif go.type == "C":
                            go_xref_obj.set_type("cellular component")
                        elif go.type == "F":
                            go_xref_obj.set_type("molecular function")
                        cross_ref_dbs.add_cross_ref_db(go_xref_obj)
                if protein.interpro:
                    self.all_db.add("InterPro")
                    for ipr in protein.interpro:
                        ipr_xref_obj = cross_ref_db(name=ipr.namespace, source="InterPro", accession_id=ipr.id,
                                                    uniprot_start=ipr.start, uniprot_end=ipr.end, provenance=ipr.provenance)
                        cross_ref_dbs.add_cross_ref_db(ipr_xref_obj)
                if protein.pfam:
                    self.all_db.add("Pfam")
                    for pfam in protein.pfam:
                        pfam_xref_obj = cross_ref_db(name=pfam.namespace, source="Pfam", accession_id=pfam.id,
                                                     uniprot_start=pfam.start, uniprot_end=pfam.end,
                                                     provenance=pfam.provenance)
                        cross_ref_dbs.add_cross_ref_db(pfam_xref_obj)
                if protein.cath:
                    self.all_db.add("CATH")
                    for cath in protein.cath:
                        cath_xref_obj = cross_ref_db(source="CATH", accession_id=cath.id, uniprot_start=cath.start,
                                                     uniprot_end=cath.end, provenance=cath.provenance)
                        cross_ref_dbs.add_cross_ref_db(cath_xref_obj)
                if protein.scop:
                    self.all_db.add("SCOP")
                    for scop in protein.scop:
                        scop_xref_obj = cross_ref_db(source="SCOP", accession_id=scop.id, uniprot_start=scop.start,
                                                     uniprot_end=scop.end, provenance=scop.provenance)
                        cross_ref_dbs.add_cross_ref_db(scop_xref_obj)
                if protein.scop2:
                    self.all_db.add("SCOP2")
                    for scop2 in protein.scop2:
                        scop2_xref_obj = cross_ref_db(source="SCOP2", accession_id=scop2.id, uniprot_start=scop2.start,
                                                      uniprot_end=scop2.end, provenance=scop2.provenance)
                        cross_ref_dbs.add_cross_ref_db(scop2_xref_obj)
                if protein.scop2B:
                    self.all_db.add("SCOP2")
                    for scop2B in protein.scop2B:
                        scop2B_xref_obj = cross_ref_db(source="SCOP2B", accession_id=scop2B.id, uniprot_start=scop2B.start,
                                                       uniprot_end=scop2B.end, provenance=scop2B.provenance)
                        cross_ref_dbs.add_cross_ref_db(scop2B_xref_obj)
                if protein.pdbekb:
                    self.all_db.add("PDBe-KB")
                    pdbekb_xref_obj = cross_ref_db(source="PDBe-KB", accession_id=protein.pdbekb.unip_id,
                                                   provenance=protein.pdbekb.provenance)
                    cross_ref_dbs.add_cross_ref_db(pdbekb_xref_obj)
                if protein.alphafold:
                    self.all_db.add("AlphaFold DB")
                    afdb_xref_obj = cross_ref_db(source="AlphaFold DB", accession_id=protein.alphafold.unip_id,
                                                 provenance=protein.alphafold.provenance)
                    cross_ref_dbs.add_cross_ref_db(afdb_xref_obj)
                if cross_ref_dbs.hasContent_():
                    macromolecule = macromoleculeType(type_="protein", id=protein.sample_id, copies=protein.sample_copies,
                                                      provenance="EMDB", name=protein.sample_name,
                                                      cross_ref_dbs=cross_ref_dbs)
                    self.macromolecules.add_macromolecule(macromolecule)

    def __read_rna(self, rna):
        if rna:
            self.all_db.add("Rfam")
            for sample_id, rfam in rna.items():
                cross_ref_dbs = cross_ref_dbsType()
                rfam_xref_obj = cross_ref_db(source="Rfam", accession_id=rfam.rfam_acc,
                                                 provenance=rfam.provenance)
                cross_ref_dbs.add_cross_ref_db(rfam_xref_obj)
                if cross_ref_dbs.hasContent_():
                    macromolecule = macromoleculeType(type_="rna", id=rfam.sample_id, copies=rfam.num_copies,
                                                      provenance="EMDB", name=rfam.sample_name,
                                                      cross_ref_dbs=cross_ref_dbs)
                    self.macromolecules.add_macromolecule(macromolecule)

    def __read_ligands(self, ligands):
        if ligands:
            for sample_id, ligand in ligands.items():
                cross_ref_dbs = cross_ref_dbsType()
                if ligand.chembl_id:
                    self.all_db.add("ChEMBL")
                    chembl_obj = cross_ref_db(source="ChEMBL", accession_id=ligand.chembl_id,
                                              provenance=ligand.provenance_chembl)
                    cross_ref_dbs.add_cross_ref_db(chembl_obj)
                if ligand.chebi_id:
                    self.all_db.add("ChEBI")
                    chebi_obj = cross_ref_db(source="ChEBI", accession_id=ligand.chebi_id,
                                             provenance=ligand.provenance_chebi)
                    cross_ref_dbs.add_cross_ref_db(chebi_obj)
                if ligand.drugbank_id:
                    self.all_db.add("DrugBank")
                    drugbank_obj = cross_ref_db(source="DrugBank", accession_id=ligand.drugbank_id,
                                                provenance=ligand.provenance_drugbank)
                    cross_ref_dbs.add_cross_ref_db(drugbank_obj)
                if cross_ref_dbs.hasContent_():
                    macromolecule = macromoleculeType(type_="ligand", id=ligand.sample_id, copies=ligand.copies,
                                                      provenance="EMDB", name=ligand.name, ccd_id=ligand.HET,
                                                      cross_ref_dbs=cross_ref_dbs)
                    self.macromolecules.add_macromolecule(macromolecule)

    def __read_complexes(self, complexes):
        if complexes:
            self.all_db.add("Complex Portal")
            for sample_id, emdb_complex in complexes.items():
                if emdb_complex:
                    cross_ref_dbs = cross_ref_dbsType()
                    if emdb_complex.cpx_list:
                        for cpx in emdb_complex.cpx_list:
                            cpx_obj = cross_ref_db(name=cpx.name, source="Complex Portal", accession_id=cpx.cpx_id,
                                                   provenance=emdb_complex.provenance,
                                                   score=round(emdb_complex.score, 2))
                            cross_ref_dbs.add_cross_ref_db(cpx_obj)
                        if cross_ref_dbs.hasContent_():
                            sample_id = emdb_complex.sample_id.split('_')[1]
                            supramolecule = supramoleculeType(type_="complex", id=sample_id,
                                                              copies=emdb_complex.sample_copies,
                                                              provenance=emdb_complex.provenance,
                                                              name=emdb_complex.name, cross_ref_dbs=cross_ref_dbs)
                            self.supramolecules.add_supramolecule(supramolecule)

    def __create_dbs(self):
        version_list = self.versions.get_all_versions()
        for database in self.all_db:
            if database in version_list:
                version = version_list[database]
                db_ver = dbType(source=database, version=version)
                self.dbs.add_db(db_ver)


