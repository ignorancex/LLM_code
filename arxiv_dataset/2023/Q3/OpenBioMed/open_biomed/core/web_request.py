from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aiohttp
import asyncio
from datetime import datetime
import json
import logging
import random
from ratelimiter import RateLimiter
import tarfile
from urllib.parse import quote

from open_biomed.data import Molecule, Protein
from open_biomed.core.tool import Tool

class Requester(Tool):
    def __init__(self) -> None:
        self.requires_async = True

class DBRequester(Requester):
    def __init__(self, db_url: str=None, timeout: int=30) -> None:
        super().__init__()
        self.db_url = db_url
        self.timeout = timeout

    @RateLimiter(max_calls=5, period=1)
    async def run(self, accession: str="") -> Any:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.db_url.format(accession=accession)) as response:
                    if response.status == 200:
                        content = await response.read()
                        content = content.decode("utf-8")
                        logging.info("Downloaded results successfully")
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Download failed. Exception: {e}")
            raise e
        return self._parse_and_save_outputs(accession, content)

    @abstractmethod
    def _parse_and_save_outputs(self, accession: str="", outputs: str="") -> Any:
        # Parse the outputs and save them at a local file
        raise NotImplementedError

class PubChemRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{accession}/SDF",
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def print_usage(self) -> str:
        query_type = ""
        if "cid" in self.db_url:
            query_type = "a PubChem ID"
        elif "name" in self.db_url:
            query_type = "molecule name"
        
        return "\n".join([
            'PubChem query.',
            'Inputs: {"accession": ' + query_type + '}',
            "Outputs: A molecule from PubChem."
        ])

    def _parse_and_save_outputs(self, accession: str="", outputs: str="") -> Tuple[List[Molecule], List[str]]:
        sdf_file = f"./tmp/pubchem_{accession}.sdf"
        with open(sdf_file, "w") as f:
            f.write(outputs)
        molecule = Molecule.from_sdf_file(sdf_file)
        return [molecule], [molecule.save_binary()]

class PubChemStructureRequester(Requester):
    def __init__(self, 
        timeout: int=30
    ) -> None:
        super().__init__()
        self.db_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{accession}/cids/JSON?Threshold={threshold}&MaxRecords={max_records}"
        self.molecule_requester = PubChemRequester()
        self.timeout = timeout

    def print_usage(self) -> str:        
        return "\n".join([
            'PubChem query.',
            'Inputs: {"accession": a molecule}',
            "Outputs: A molecule from PubChem."
        ])

    @RateLimiter(max_calls=5, period=1)
    async def run(self, molecule: Molecule=None, threshold: float=0.8, max_records=10) -> Tuple[List[Molecule], List[str]]:
        molecule._add_smiles()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                url = self.db_url.format(accession=molecule.smiles, threshold=int(threshold * 100), max_records=max_records)
                async with session.get(url.replace("#", "%23")) as response:
                    if response.status == 200:
                        content = await response.read()
                        content = json.loads(content.decode("utf-8"))
                        logging.info("Downloaded results successfully")
                    elif response.status == 404:
                        logging.info("No similar structures found!")
                        return [molecule], [molecule.save_binary()]
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Download failed. Exception: {e}")
            raise e
        all_mols, all_files = [], []
        for cid in content['IdentifierList']['CID']:
            mol, mol_file = await self.molecule_requester.run(cid)
            all_mols.extend(mol)
            all_files.extend(mol_file)
        return all_mols, all_files

class ChemBLRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://www.ebi.ac.uk/chembl/api/data/molecule?molecule_chembl_id={accession}&format=json", 
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def _parse_and_save_outputs(self, accession: str="", outputs: str="") -> str:
        obj = json.loads(outputs)
        sdf_file = f"./tmp/chembl_{accession}.sdf"
        with open(sdf_file, "w") as f:
            f.write(obj["molecules"][0]["molecule_structures"]["molfile"])
        molecule = Molecule.from_sdf_file(sdf_file)
        return [molecule], [molecule.save_binary()]

class UniProtRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://rest.uniprot.org/uniprotkb/{accession}?format=json", 
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def print_usage(self) -> str:
        return "\n".join([
            'UniProt query.',
            'Inputs: {"accession": a UniProt ID}',
            "Outputs: A protein from UniProt."
        ])

    def _parse_and_save_outputs(self, accession: str="", outputs: str="") -> str:
        obj = json.loads(outputs)
        protein = Protein.from_fasta(obj["sequence"]["value"])
        protein.name = f"uniprot_{accession}"
        return [protein], [protein.save_binary()]

class PDBRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://files.rcsb.org/download/{accession}.pdb", 
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def print_usage(self) -> str:
        database = "AlphaFoldDB" if "alphafold" in self.db_url else "PDB"
        return "\n".join([
            'PDB structure query.',
            'Inputs: {"accession": a ' + database + ' ID}',
            "Outputs: A protein from the database."
        ])

    def _parse_and_save_outputs(self, accession: str="", outputs: str="") -> str:
        pdb_file = f"./tmp/pdb_{accession}.pdb"
        with open(pdb_file, "w") as f:
            f.write(outputs)
        protein = Protein.from_pdb_file(pdb_file)
        return [protein], [protein.save_binary()]

class WebSearchRequester(Tool):
    def __init__(self, timeout: int=30) -> None:
        self.timeout = timeout

    def print_usage(self) -> str:
        return "\n".join([
            'Web search.',
            'Inputs: {"query": Any query inputs}',
            "Outputs: results from the search engine."
        ])

    def run(self, query: str) -> Tuple[List[str], List[str]]:

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 1234567890'
        }
        query_url = "http://pre.chatdd.pharmolix.com/v2/api/deepinsight/generate_query"
        data = {
            'chat_session_id': "FwAalhadkajhddkaadfwes",
            'action': False,
            'chat_messages': [
                {"role": "user", "content": f'<p>{query}</p><p><br></p>'}
            ]
        }
        response = requests.post(query_url, headers=headers, json=data)
        question = response.json()

        rag_url = "http://101.200.137.30:1112/rag/v1/common_rag"

        if not question['query_list']:
            question['query_list'] = []
        question["recall_params"] = {
            "PaperDB": [
                "meeting",
                "pubmed_abstract",
                "pubmed_full_text"
            ],
            "NewsDB": [
                "press",
                "media",
                "wechat",
                "wechat_realtime",
                "press_realtime"
            ],
            "WebSearch": None,
            "Clinicaltrial_DB": [
                "clinicaltrials"
            ],
            "Policy_DB": [
                "policy"
            ],
            "Principle_DB": [
                "principle"
            ],
            "PatentLaw_DB": [
                "patentlaw"
            ]
        }
        question["top_k"] = 5
        res = requests.post(rag_url, json=question)
        #result = {"query": [query] + question['query_list'],
        #          "result": res.json()["data"]}
        #result = {"result": [i["text"] for i in res.json()["data"]]}
        result = "\n\n\n".join([i["text"] for i in res.json()["data"]])
        return [result], [result]


class MMSeqsRequester(Requester):
    def __init__(self, 
        host: str="https://api.colabfold.com/", 
        job_url_suffix: str="",
        timeout: int=30
    ) -> None:
        super().__init__()
        self.host = host
        self.job_url_suffix = job_url_suffix
        self.timeout = timeout

    @RateLimiter(max_calls=5, period=1)
    async def submit_job(self, data: Dict[str, Any]) -> str:
        content = {"status": "UNKNOWN"}
        while True:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        url=f"{self.host}/ticket{self.job_url_suffix}",
                        data=data,
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            content = json.loads(content.decode("utf-8"))
                            if not content["status"] in ["UNKNOWN", "RATELIMIT"]:
                                break
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception()
                await asyncio.sleep(5 + random.randint(0, 5))
            except Exception as e:
                content = None
                logging.error(f"Web request failed. Exception: {e}")
                raise e
        
        if content["status"] == "ERROR":
            raise Exception(f'Web API is giving errors. Please confirm your input is valid. If error persists, please try again an hour later.')

        if content["status"] == "MAINTENANCE":
            raise Exception(f'Web API is undergoing maintenance. Please try again in a few minutes.')

        return content["id"]

    @RateLimiter(max_calls=5, period=1)
    async def wait_finish(self, id: str="") -> str:
        content = {"status": "UNKNOWN"}
        time_elapsed = 0
        while True:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.get(
                        url=f"{self.host}/ticket/{id}",
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            content = json.loads(content.decode("utf-8"))
                            if not content["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                                break
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception()
                t = 5 + random.randint(0, 5)
                time_elapsed += t
                logging.info(f"Current job status: {content['status']}, {time_elapsed} seconds elapsed.")
                await asyncio.sleep(t)
            except Exception as e:
                content = None
                logging.error(f"Web request failed. Exception: {e}")
                raise e
        return content["status"]
    
    @RateLimiter(max_calls=5, period=1)
    async def download(self, id: str="") -> str:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(
                    url=f"{self.host}/result/download/{id}",
                ) as response:
                    if response.status == 200:
                        content = await response.read()
                        return content
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Web request failed. Exception: {e}")
            raise e

class MSARequester(MMSeqsRequester):
    def __init__(self, 
        host: str="https://api.colabfold.com", 
        mode: str="all",
        timeout: int=30
    ) -> None:
        super().__init__(host=host, job_url_suffix="/msa", timeout=timeout)
        self.mode = mode

    def print_usage(self) -> str:
        return "\n".join([
            'Multiple sequence alignment.',
            'Inputs: {"protein": a protein sequence}',
            "Outputs: A .a3m file comprising metadata of similar sequences."
        ])

    async def run(self, protein: Protein="") -> Tuple[List[str], List[str]]:
        fasta = f">1\n{protein.sequence}\n"
        data = {
            "q": fasta,
            "mode": self.mode,
        }
        while True:
            id = await self.submit_job(data)
            logging.info(f"Request id: {id}")
            status = await self.wait_finish(id)
            if status == "COMPLETE":
                break
        content = await self.download(id)
        timestamp = int(datetime.now().timestamp() * 1000)
        tar_file = f"./tmp/msa_results_{timestamp}.tar.gz"
        with open(tar_file, "wb") as f: f.write(content)
        logging.info(f"File saved at {tar_file}")
        with tarfile.open(tar_file) as tar_gz:
            folder_name = tar_file.rstrip(".tar.gz")
            os.makedirs(folder_name, exist_ok=True)
            tar_gz.extractall(folder_name)
        ret = f"{folder_name}/uniref.a3m"
        return [ret], [ret]

class FoldSeekRequester(MMSeqsRequester):
    def __init__(self, 
        host: str="https://search.foldseek.com/api", 
        mode: str="3diaa",
        database: List[str]=["BFVD", "afdb50", "afdb-swissprot", "afdb-proteome", "bfmd", "cath50", "mgnify_esm30", "pdb100", "gmgcl_id"],
        timeout: int=60
    ) -> None:
        super().__init__(host, "", timeout)
        self.mode = mode
        self.database = database

    def print_usage(self) -> str:
        return "\n".join([
            'Foldseek.',
            'Inputs: {"protein": a protein backbone structure (typically in pdb format)}',
            "Outputs: A .m8 file comprising metadata of similar structures."
        ])

    async def run(self, protein: Protein="") -> Tuple[List[str], List[str]]:
        timestamp = int(datetime.now().timestamp() * 1000)
        pdb_file = f"./tmp/protein_{timestamp}.pdb"
        protein.save_pdb(pdb_file)
        form_data = aiohttp.FormData()
        form_data.add_field("mode", self.mode)
        for db in self.database:
            form_data.add_field("database[]", db)
        # Add the file field (open file in binary mode)
        f = open(pdb_file, 'rb')
        form_data.add_field('q', f, filename=pdb_file, content_type='application/octet-stream')

        try:
            while True:
                id = await self.submit_job(form_data)
                logging.info(f"Request id: {id}")
                status = await self.wait_finish(id)
                if status == "COMPLETE":
                    logging.info("Task completed. Try downloading...")
                    break
            content = await self.download(id)
        finally:
            f.close()
        tar_file = f"./tmp/foldseek_results_{timestamp}.tar.gz"
        with open(tar_file, "wb") as f: f.write(content)
        logging.info(f"File saved at {tar_file}")
        with tarfile.open(tar_file) as tar_gz:
            folder_name = tar_file.rstrip(".tar.gz")
            os.makedirs(folder_name, exist_ok=True)
            tar_gz.extractall(folder_name)
        ret = f"{folder_name}"
        return [ret], [ret]

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    """
    requester = UniProtRequester()
    asyncio.run(requester.run("P0DTC2"))

    requester = PDBRequester()
    asyncio.run(requester.run("6LVN"))
    
    requester = PubChemRequester()
    asyncio.run(requester.run("240"))

    requester = ChemBLRequester()
    asyncio.run(requester.run("CHEMBL941"))

    requester = MSARequester()
    asyncio.run(requester.run(Protein.from_binary_file("./tmp/uniprot_P0DTC2.pkl")))
    #asyncio.run(requester.run(Protein.from_fasta("MMVEVRFFGPIKEENFFIKANDLKELRAILQEKEGLKEWLGVCAIALNDHLIDNLNTPLKDGDVISLLPPVCGG")))
    requester = FoldSeekRequester(database=["afdb50"])
    asyncio.run(requester.run(Protein.from_pdb_file("./tmp/demo/foldseek.pdb")))

    requester = PDBRequester("https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb")
    asyncio.run(requester.run("A0A2E8J446"))

    websearchrequester = WebSearchRequester()
    qurey = "Please tell me something about diabetes"  
    print(websearchrequester.run(qurey))

    requester = PubChemRequester(db_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF")
    asyncio.run(requester.run("dimethoxy-sulfanylidene-(3,5,6-trichloropyridin-2-yl)oxy-lambda5-phosphane"))
    """

    requester = PubChemStructureRequester()
    asyncio.run(requester.run(Molecule.from_smiles("CN1CCC[C@H]1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN([C@H](C6)CC#N)C(=O)C(=C)F"), threshold=0.8))
