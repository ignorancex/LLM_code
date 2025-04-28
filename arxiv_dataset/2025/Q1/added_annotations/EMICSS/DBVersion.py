import re
import json
from datetime import date
from datetime import timedelta
import requests
import lxml.etree as ET


class Version:
    """
    Database versions for all mapped external resources
    """
    def __init__(self):
        self.emdb = None
        self.pdbe = None
        self.drugbank = self.__find_db_version()
        self.interpro, self.pfam = self.__find_ipr_pfam_version()
        self.cath = self.__find_cath_version()
        self.chembl = self.__find_chembl_version()
        self.go = self.__find_go_version()
        self.uniprot = self.__find_unp_version()
        self.alphafold = self.__find_afdb_version()  # TODO: Version is currently fixed to all entries
        self.scop = "1.7.5" # TODO: Create methods to obtain version
        self.scop2 = "2.0" # TODO: Create methods to obtain version
        self.rfam = self.__find_rfam_version()
        self.cpx = None  # TODO: Add version
        self.scop2b = None # TODO: Add version
        self.chebi = None # TODO: Add version
        self.empiar = None # TODO: Add version
        self.pdbekb = None # TODO: Add version
        self.pubmed = None # TODO: Add version
        self.pmc = None # TODO: Add version
        self.today = date.today().strftime("%Y-%m-%d")

    def get_all_versions(self):
        return {
            'EMDB': self.emdb,
            'EMPIAR': self.empiar,
            'PDBe': self.pdbe,
            'PDBe-KB': self.pdbekb,
            'Complex Portal': self.cpx,
            'ChEMBL': self.chembl,
            'ChEBI': self.chebi,
            'DrugBank': self.drugbank,
            'UniProt': self.uniprot,
            'GO': self.go,
            'Pfam': self.pfam,
            'InterPro': self.interpro,
            'CATH': self.cath,
            'SCOP': self.scop,
            'SCOP2': self.scop2,
            'SCOP2B':self.scop2b,
            'AlphaFold DB': self.alphafold,
            'Rfam': self.rfam
        }

    def __find_rfam_version(self):
        url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/README"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                lines = response.text.split('\n')
                # Find the first line containing the word "Release"
                for line in lines:
                    if "Release" in line:
                        match = re.search(r'Release (\d+\.\d+)', line)
                        if match:
                            return match.group(1)
            return None

    def __find_afdb_version(self):
        url = "https://alphafold.ebi.ac.uk/api/prediction/Q5VSL9"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                res_text = response.text
                data = json.loads(res_text)
                if data:
                    return data[0]["latestVersion"]
            return None


    def __find_cpx_version(self):
        url = "http://ftp.ebi.ac.uk/pub/databases/intact/complex/current/"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return ""
        else:
            if response.status_code == 200 and response.content:
                html = response.content.decode('utf-8')
                cpx_ver = re.findall("\d*\-\w*\-\d*", html)[0]
                return cpx_ver
            return None

    def __find_db_version(self):
        url = "https://go.drugbank.com/releases/latest"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                html = response.content.decode('utf-8')
                vers = re.findall("<td>\d*\.\d*\.\d*", html)[0]
                drugbank_ver = vers.split(">")[1]
                return drugbank_ver
            return None

    def __find_ipr_pfam_version(self):
        url = f"https://www.ebi.ac.uk/interpro/api/"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None, None
        else:
            if response.status_code == 200:
                res_text = response.text
                data = json.loads(res_text)
                if 'databases' in data:
                    interpro_ver = data['databases']['interpro']['version']
                    pfam_ver = data['databases']['pfam']['version']
                    return interpro_ver, pfam_ver
            return None, None

    def __find_cath_version(self):
        url = "https://www.cathdb.info/"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.SSLError) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                html = response.content.decode('utf-8')
                vers = re.findall("<h1>CATH / Gene3D <small>\w*\d*\.\d*", html)[0]
                cath_ver = vers.split("small>")[1]
                return cath_ver
            return None

    def __find_chembl_version(self):
        url = f"https://www.ebi.ac.uk/chembl/api/data/status/"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                root = ET.fromstring(response.content)
                for x in list(root.iter('response')):
                    return x.find('chembl_db_version').text
            return None

    def __find_go_version(self):
        url = "http://current.geneontology.org/release_stats/go-stats-summary.json"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                data = json.loads(response.content)
                return data['release_date']
            return None

    def __find_unp_version(self):
        url = "https://ftp.uniprot.org/pub/databases/uniprot/relnotes.txt"
        try:
            response = requests.get(url, timeout=10)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            return None
        else:
            if response.status_code == 200 and response.content:
                return response.text.split("\n")[0]
            return None
        
