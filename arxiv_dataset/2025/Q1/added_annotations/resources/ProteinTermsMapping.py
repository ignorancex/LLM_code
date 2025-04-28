import requests, re, gzip
from models import GO, Interpro, Pfam, Cath, SCOP, SCOP2, SCOP2B, Pdbekb
import lxml.etree as ET
from Bio import Align
import copy
import xmltodict

class ProteinTermsMapping:
    """
    Extract GO, InterPro, Pfam, CATH, SCOP, SCOP2 and SCOP2B terms from SIFTS

    If sequence + model => Fetch from sifts
    """

    def __init__(self, proteins, sifts_prefix, is_go=True, is_interpro=True, is_pfam=True, is_cath=True,
                 is_scop=True, is_scop2=True, is_scop2B=True, is_pdbekb=True):
        self.proteins = proteins

        self.is_interpro = is_interpro
        self.is_go = is_go
        self.is_pfam = is_pfam
        self.is_cath = is_cath
        self.is_scop = is_scop
        self.is_scop2 = is_scop2
        self.is_scop2B = is_scop2B
        self.is_pdbekb = is_pdbekb
        self.sifts_prefix = sifts_prefix

    def execute(self, uniprot_with_models):
        for protein in self.proteins:
            if protein.uniprot_id:
                if self.is_pdbekb:
                    if protein.uniprot_id in uniprot_with_models:
                        pdbekb = Pdbekb(protein.uniprot_id, "UniProt")
                        protein.pdbekb = pdbekb

                if protein.sequence:
                    map_sequence = self.strip_sequence(protein.sequence)
                    if map_sequence:
                        protein_sequence, go_data, ipr_data, pf_data = self.fetch_uniprot(protein.uniprot_id)
                        if protein_sequence:
                            aligned_positions, score = self.align(protein_sequence, map_sequence)

                            if score > 0.5 * min([len(protein_sequence), len(map_sequence)]):
                                if protein.pdb:
                                    go_matches, ipr_matches, pfam_matches, cath_matches, scop_matches, scop2_matches, scop2B_matches = self.parse_sifts(
                                        protein, aligned_positions)

                                    ipr_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(ipr_matches, aligned_positions))
                                    pfam_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(pfam_matches, aligned_positions))
                                    cath_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(cath_matches, aligned_positions))
                                    scop_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(scop_matches, aligned_positions))
                                    scop2_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(scop2_matches, aligned_positions))
                                    scop2B_matches = self.remove_duplications(
                                        self.uniprot_to_map_positions(scop2B_matches, aligned_positions))

                                    for go_id in go_matches:
                                        if go_id in go_data:
                                            go = copy.deepcopy(go_data[go_id])
                                            go.provenance = "PDBe"
                                            protein.go.add(go)
                                    for ipr_id, start, end, unp_start, unp_end in ipr_matches:
                                        if ipr_id in ipr_data:
                                            ipr = copy.deepcopy(ipr_data[ipr_id])
                                            ipr.provenance = "PDBe"
                                            ipr.start = start
                                            ipr.end = end
                                            ipr.unp_start = unp_start
                                            ipr.unp_end = unp_end
                                            protein.interpro.add(ipr)
                                    for pf_id, start, end, unp_start, unp_end in pfam_matches:
                                        if pf_id in pf_data:
                                            pfam = copy.deepcopy(pf_data[pf_id])
                                            pfam.provenance = "PDBe"
                                            pfam.start = start
                                            pfam.end = end
                                            pfam.unp_start = unp_start
                                            pfam.unp_end = unp_end
                                            protein.pfam.add(pfam)
                                    for cath_id, start, end, unp_start, unp_end in cath_matches:
                                        cath = Cath()
                                        cath.id = cath_id
                                        cath.start = start
                                        cath.end = end
                                        cath.unp_start = unp_start
                                        cath.unp_end = unp_end
                                        cath.unip_id = protein.uniprot_id
                                        cath.provenance = "PDBe"
                                        protein.cath.add(cath)
                                    for scop_id, start, end, unp_start, unp_end in scop_matches:
                                        scop = SCOP()
                                        scop.id = scop_id
                                        scop.start = start
                                        scop.end = end
                                        scop.unp_start = unp_start
                                        scop.unp_end = unp_end
                                        scop.unip_id = protein.uniprot_id
                                        scop.provenance = "PDBe"
                                        protein.scop.add(scop)
                                    for scop2_id, start, end, unp_start, unp_end in scop2_matches:
                                        scop2 = SCOP2()
                                        scop2.id = scop2_id
                                        scop2.start = start
                                        scop2.end = end
                                        scop2.unp_start = unp_start
                                        scop2.unp_end = unp_end
                                        scop2.unip_id = protein.uniprot_id
                                        scop2.provenance = "PDBe"
                                        protein.scop2.add(scop2)
                                    for scop2B_id, start, end, unp_start, unp_end in scop2B_matches:
                                        scop2B = SCOP2B()
                                        scop2B.id = scop2B_id
                                        scop2B.start = start
                                        scop2B.end = end
                                        scop2B.unp_start = unp_start
                                        scop2B.unp_end = unp_end
                                        scop2B.unip_id = protein.uniprot_id
                                        scop2B.provenance = "PDBe"
                                        protein.scop2B.add(scop2B)

        return self.proteins


    def remove_duplications(self, matches):
        refs = {}

        for ref_id, start, end, unp_start, unp_end in matches:
            if ref_id in refs:
                if start < refs[ref_id][1]:
                    refs[ref_id][1] = start
                    refs[ref_id][3] = unp_start
                if end > refs[ref_id][2]:
                    refs[ref_id][2] = end
                    refs[ref_id][4] = unp_end
            else:
                refs[ref_id] = [ref_id, start, end, unp_start, unp_end]

        return list(refs.values())

    def extract_uniprot_position(self, db_tag, segment,
                                 namespaces={'x': 'http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd'}):
        region = db_tag.getparent()
        pdb_start = region.get("start")
        pdb_end = region.get("end")
        start_tag = segment.xpath(f".//x:residue[@dbResNum='{pdb_start}']/x:crossRefDb[@dbSource='UniProt']/@dbResNum",
                                  namespaces=namespaces)
        end_tag = segment.xpath(f".//x:residue[@dbResNum='{pdb_end}']/x:crossRefDb[@dbSource='UniProt']/@dbResNum",
                                namespaces=namespaces)
        start = int(start_tag[0]) if len(start_tag) > 0 else None
        end = int(end_tag[0]) if len(end_tag) > 0 else None
        return start, end

    def convert_positions(self, position, positions):
        last_i = len(positions[0]) - 1
        for i, (x, y) in enumerate(positions[0]):
            if int(position) < int(x):
                if i == 0:
                    return positions[1][0][0]
                else:
                    return positions[1][i - 1][1]
            elif int(position) > y and i == last_i:
                return positions[1][last_i][1]
            elif int(position) <= y:
                j = int(position) - x
                return positions[1][i][0] + j

    def uniprot_to_map_positions(self, dataset, positions):
        new_dataset = set()
        unp_positions, map_positions = positions
        for ref, unp_start, unp_end in dataset:
            start = self.convert_positions(unp_start, positions)
            end = self.convert_positions(unp_end, positions)

            # Miniumum 15 amino acids coverage to avoid mapping of bad alignment fragments
            if start > 0 and end > 0 and end - start > 15:
                if start < end:
                    new_dataset.add((ref, start, end, unp_start, unp_end))
        return new_dataset

    def parse_sifts(self, protein, positions):
        models = protein.pdb
        unp_id = protein.uniprot_id
        go_matches = set()
        ipr_matches = set()
        pfam_matches = set()
        cath_matches = set()
        scop_matches = set()
        scop2_matches = set()
        scop2B_matches = set()
        unp_positions = positions[0]
        map_positions = positions[1]
        unp_begin = unp_positions[0][0] + 1
        unp_end = unp_positions[-1][1] + 1

        for model in models:
            pdb_id = model.pdb_id

            try:
                file = gzip.open(self.sifts_prefix + pdb_id[1:3] + f"/{pdb_id}.xml.gz", "rb")
                root = ET.fromstring(file.read())
                namespaces = {'x': 'http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd'}
                segments = root.xpath("//x:segment", namespaces=namespaces)

                for segment in segments:
                    unp_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='UniProt']", namespaces=namespaces)
                    for unp in unp_list:
                        if unp.get("dbAccessionId") == unp_id:
                            if unp_end >= int(unp.get("start")) and unp_begin <= int(unp.get("end")):
                                if self.is_go:
                                    go_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='GO']",
                                                            namespaces=namespaces)
                                    for go in go_list:
                                        go_id = go.get("dbAccessionId")
                                        go_matches.add(go_id)
                                if self.is_interpro:
                                    interpro_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='InterPro']",
                                                                  namespaces=namespaces)
                                    for ipr in interpro_list:
                                        ipr_id = ipr.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(ipr, segment)
                                        if start and end:
                                            ipr_matches.add((ipr_id, start, end))
                                if self.is_pfam:
                                    pfam_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='Pfam']",
                                                              namespaces=namespaces)
                                    for pfam in pfam_list:
                                        pfam_id = pfam.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(pfam, segment)
                                        if start and end:
                                            pfam_matches.add((pfam_id, start, end))
                                if self.is_cath:
                                    cath_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='CATH']",
                                                              namespaces=namespaces)
                                    for cath in cath_list:
                                        cath_id = cath.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(cath, segment)
                                        if start and end:
                                            cath_matches.add((cath_id, start, end))
                                if self.is_scop:
                                    scop_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='SCOP']",
                                                              namespaces=namespaces)
                                    for scop in scop_list:
                                        scop_id = scop.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(scop, segment)
                                        if start and end:
                                            scop_matches.add((scop_id, start, end))
                                if self.is_scop2:
                                    scop2_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='SCOP2']",
                                                               namespaces=namespaces)
                                    for scop2 in scop2_list:
                                        scop2_id = scop2.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(scop2, segment)
                                        if start and end:
                                            scop2_matches.add((scop2_id, start, end))
                                if self.is_scop2B:
                                    scop2B_list = segment.xpath(".//x:mapRegion/x:db[@dbSource='SCOP2B']",
                                                                namespaces=namespaces)
                                    for scop2B in scop2B_list:
                                        scop2B_id = scop2B.get("dbAccessionId")
                                        start, end = self.extract_uniprot_position(scop2B, segment)
                                        if start and end:
                                            scop2B_matches.add((scop2B_id, start, end))
            except (FileNotFoundError, ET.XMLSyntaxError):
                continue
        return go_matches, ipr_matches, pfam_matches, cath_matches, scop_matches, scop2_matches, scop2B_matches

    def pfam_api_maponly(self, protein):
        """
        Fetching Pfam ID, start and end position from PFAM API for map only entries
        """
        unp_id = protein.uniprot_id
        pfam_map_matches = set()

        if self.is_pfam:
            url = f"https://pfam.xfam.org/protein/{unp_id}?output=xml"
            response = requests.get(url)
            if response.status_code == 200 and response.content:
                data = xmltodict.parse(response.text)
                data = data['pfam']['entry']
                if 'matches' in data:
                    matches = data['matches']['match']
                    if '@accession' in matches:
                        matches = [matches]
                    for match in matches:
                        pfam_id = match['@accession']
                        start = int(match['location']['@start'])
                        end = int(match['location']['@end'])
                        pfam_map_matches.add((pfam_id, start, end))

        return pfam_map_matches

    def strip_sequence(self, sequence):
        """
        Strip UNK and spaces from the sequence
        """
        sequence = re.sub('\s', '', sequence)
        sequence = re.sub('\(.*?\)', 'X', sequence)
        return sequence

    def align(self, target, query):
        """
        Align the map sequence and target full protein sequence to return
        mapped positions (indexes: beggining with 0).
        i.e. (((2, 13), (38, 46)), ((0, 11), (11, 19)))
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = -2
        alignments = aligner.align(target, query)
        try:
            optimal = next(alignments)
        except StopIteration:
            return (None, None), 0

        return optimal.aligned, optimal.score

    def fetch_uniprot(self, uid):
        sequence = ""
        go_data = {}
        interpro_data = {}
        pfam_data = {}
        url = f"https://www.uniprot.org/uniprotkb/{uid}.xml"
        namespace = {'x': 'http://uniprot.org/uniprot'}
        response = requests.get(url)
        if response.status_code == 200 and response.content:
            try:
                root = ET.fromstring(response.content)
                sequence_tag = root.xpath("//x:entry/x:sequence", namespaces=namespace)
                if sequence_tag:
                    sequence = sequence_tag[0].text

                    if self.is_go:
                        go_elements = root.findall(".//{http://uniprot.org/uniprot}dbReference[@type='GO']")
                        for element in go_elements:
                            go = GO()
                            go.id = element.get("id")
                            go.unip_id = uid
                            terms = element.findall("{http://uniprot.org/uniprot}property[@type='term']")
                            if terms:
                                term_text = terms[0].get("value")
                                go.type = term_text[0]
                                go.namespace = term_text[2:]
                                go_data[go.id] = go
                    if self.is_interpro:
                        interpro_elements = root.findall(".//{http://uniprot.org/uniprot}dbReference[@type='InterPro']")
                        for element in interpro_elements:
                            interpro = Interpro()
                            interpro.id = element.get("id")
                            interpro.unip_id = uid
                            terms = element.findall("{http://uniprot.org/uniprot}property[@type='entry name']")
                            if terms:
                                interpro.namespace = terms[0].get("value")
                                interpro_data[interpro.id] = interpro
                    if self.is_pfam:
                        pfam_elements = root.findall(".//{http://uniprot.org/uniprot}dbReference[@type='Pfam']")
                        for element in pfam_elements:
                            pfam = Pfam()
                            pfam.id = element.get("id")
                            pfam.unip_id = uid
                            terms = element.findall("{http://uniprot.org/uniprot}property[@type='entry name']")
                            if terms:
                                pfam.namespace = terms[0].get("value")
                                pfam_data[pfam.id] = pfam
            except ET.XMLSyntaxError:
                print(f"Failed to parse {uid}")

        return sequence, go_data, interpro_data, pfam_data

    def export_tsv(self, go_logger, interpro_logger, pfam_logger, cath_logger, scop_logger, scop2_logger, scop2B_logger,
                   pdbekb_logger):
        for protein in self.proteins:
            if self.is_go and protein.go:
                for go in protein.go:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{go.id}\t{go.namespace}\t{go.type}\t{go.provenance}"
                    go_logger.info(row)
            if self.is_interpro and protein.interpro:
                for ipr in protein.interpro:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{ipr.id}\t{ipr.namespace}\t{ipr.start}\t{ipr.end}" \
                          f"\t{ipr.unp_start}\t{ipr.unp_end}\t{ipr.provenance}"
                    interpro_logger.info(row)
            if self.is_pfam and protein.pfam:
                for pfam in protein.pfam:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{pfam.id}\t{pfam.namespace}\t{pfam.start}\t{pfam.end}" \
                          f"\t{pfam.unp_start}\t{pfam.unp_end}\t{pfam.provenance}"
                    pfam_logger.info(row)
            if self.is_cath and protein.cath:
                for cath in protein.cath:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{cath.id}\t{cath.start}\t{cath.end}\t{cath.unp_start}" \
                          f"\t{cath.unp_end}\t{cath.provenance}"
                    cath_logger.info(row)
            if self.is_scop and protein.scop:
                for scop in protein.scop:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{scop.id}\t{scop.start}\t{scop.end}\t{scop.unp_start}" \
                          f"\t{scop.unp_end}\t{scop.provenance}"
                    scop_logger.info(row)
            if self.is_scop2 and protein.scop2:
                for scop2 in protein.scop2:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{scop2.id}\t{scop2.start}\t{scop2.end}\t{scop2.unp_start}" \
                          f"\t{scop2.unp_end}\t{scop2.provenance}"
                    scop2_logger.info(row)
            if self.is_scop2B and protein.scop2B:
                for scop2B in protein.scop2B:
                    row = f"{protein.emdb_id}\t{protein.sample_id}\t{scop2B.id}\t{scop2B.start}\t{scop2B.end}\t{scop2B.unp_start}" \
                          f"\t{scop2B.unp_end}\t{scop2B.provenance}"
                    scop2B_logger.info(row)
            if self.is_pdbekb and protein.pdbekb:
                row = f"{protein.emdb_id}\t{protein.sample_id}\t{protein.uniprot_id}\tUniProtKB"
                pdbekb_logger.info(row)
