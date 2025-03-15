from autoqchem.slurm_manager import slurm_manager
from autoqchem.helper_classes import slurm_status
from autoqchem.helper_classes import slurm_job
from autoqchem.gaussian_log_extractor import *
from rdkit import Chem
import os
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scripts.generate_smiles import *

from prompt_toolkit import prompt

file_name=input('folder name of remote directory: ')
user = input('user: ')
host = input('host: ')
port = input('port: ')

sm=slurm_manager(user=user, host=host, file_name=file_name, port = port)

data_directory = sm.workdir
resource_block = f"%CPU=\n"
task = f"M062X/6-31G(d) scf=(xqc,tight) integral(grid=ultrafinegrid)"
charge = '0'
multiplicity = '1'

def create_reorg_geom_input_files(smi) -> None:
    job = list(sm.get_jobs(can = smi).items())[0][1]
    directory = job.directory
   

    name = f"{job.inchikey}_pyramid"
    
    mol = Chem.MolFromSmiles(smi)
    mol_F = get_fluoride_adduct([mol])[0]
    smi_F = Chem.MolToSmiles(mol_F)
    
    job_F = list(sm.get_jobs(can = smi_F).items())[0][1]
    job_F_log = f"{job_F.directory}/{job_F.base_name}.log"
    le = gaussian_log_extractor(job_F_log)
    le.get_atom_labels()
    le.get_geometry()
    coords = le.geom[list('XYZ')].applymap(lambda x: f"{x:.6f}")
    coords.insert(0, 'Atom', le.labels)
    coords_block = "\n".join(map(" ".join, coords.values)) + "\n\n"
    new_coords_block = re.sub('F.*\\n','', coords_block) 
    generate_pyram_gaussian_input_file(task, name, directory, resource_block, new_coords_block, charge, multiplicity)
    sm._create_slurm_file_from_gaussian_file(name, directory, wall_time='20:00:00')
    

def generate_pyram_gaussian_input_file(task, name, directory, resource_block, coords_block, charge, multiplicity) -> None:

    output = ""
    output += resource_block
    output += f"%RWF={name}.rwf\n"
    output += f"%NoSave\n"
    output += f"%Chk={name}.chk\n"
    output += f"# {task}\n\n"
    output += f"{name}\n\n"
    output += f"{charge} {multiplicity}\n"
    output += f"{coords_block.strip()}\n"
    output += f"\n"
     
    output += f"\n\n"

    file_path = f"{directory}/{name}.com"
    with open(file_path, "w") as file:
        file.write(output)

    logger.debug(f"Generated a Gaussian pyramidalized input file in {file_path}")
    
def submit_pyramid_spe_calc(smi):
    sm.connect()
    job = list(sm.get_jobs(can = smi).items())[0][1]
    directory = job.directory
    remote_dir = job.remote_dir
    print(remote_dir)
    try : 
        sm.connection.run(f"mkdir {remote_dir}")
    except : pass
    sm.connection.put(f"{job.directory}/{job.inchikey}_pyramid.sh", remote_dir)
    sm.connection.put(f"{job.directory}/{job.inchikey}_pyramid.com", remote_dir)
    with sm.connection.cd(remote_dir):
        sm.connection.run(f"\n sbatch {job.inchikey}_pyramid.sh")

def retrieve_pyramid_log_file(smi):
    sm.connect()
    job = list(sm.get_jobs(can = smi).items())[0][1]
    try :
        sm.connection.run(f"grep 'Normal termination' {job.remote_dir}/{job.inchikey}_pyramid.log")     
    except :
        print(f"job {job.inchikey} has not terminated normally")
    log_file = sm.connection.get(f"{job.remote_dir}/{job.inchikey}_pyramid.log", local=f"{job.directory}/{job.inchikey}_pyramid.log")  
    

def reorganization_nrj(smiles):
    os.chdir(data_directory)
    reorg_nrjs = []
    for smi in smiles : 
        fails = 0
        job = sm.get_jobs(can = smi)
        inchikey = list(job.values())[0].inchikey
       
        os.chdir(list(job.values())[0].directory)
        try :
            pyramid_nrj = get_SCF_energy_pyramid(inchikey)
        except :
            #print("no pyram nrj found")
            fails += 1
        try :
            opt_nrj = get_SCF_energy_opt(inchikey)
        except :
            #print("no opt nrj found")
            fails += 1
        reorg_nrj = (pyramid_nrj - opt_nrj)*2625.5002
        #print(fails)
        if fails >= 1 :
            reorg_nrjs.append(None)
        else :
            reorg_nrjs.append(reorg_nrj)
    return(reorg_nrjs)


def get_SCF_energy_opt(inchikey):
    patternSCF = ".*SCF\sDone:\s{1,}E\S{1,}\s=\s{1,}\S{1,}\s{1,}A.U.\safter\s{1,}\d{1,}\scycles.*"
    with open(f"{inchikey}_conf_0.log", 'r') as f:
        lines = f.readlines()    
    SCF = []
    for line in lines :        
        resultSCF = re.match(patternSCF, line)        
        if resultSCF is not None : 
            resultSCF = resultSCF.string
            value = float(find_float(resultSCF)[0])
            SCF.append(value)
    return(SCF[-1])


def get_SCF_energy_pyramid(inchikey):
    patternSCF = ".*SCF\sDone:\s{1,}E\S{1,}\s=\s{1,}\S{1,}\s{1,}A.U.\safter\s{1,}\d{1,}\scycles.*"
    with open(f"{inchikey}_pyramid.log", 'r') as f:
        lines = f.readlines()    
    SCF = []
    for line in lines :        
        resultSCF = re.match(patternSCF, line)        
        if resultSCF is not None : 
            resultSCF = resultSCF.string
            value = float(find_float(resultSCF)[0])
            SCF.append(value)
    return(SCF[-1])

def find_float(string) : 
    L = re.findall("-{0,1}\d{1,}\.\d{1,}", string)
    return L