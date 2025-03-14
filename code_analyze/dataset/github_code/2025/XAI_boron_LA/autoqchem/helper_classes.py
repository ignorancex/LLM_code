import enum
import os
import numpy as np
from dataclasses import dataclass

import yaml

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yml")))
k_in_kcal_per_mol_K = 0.0019872041
Hartree_in_kcal_per_mol = 627.5
T = 298


@enum.unique
class slurm_status(enum.IntEnum):
    """Slurm job status enumerator."""

    created = 1  #: job files have been created and stored locally
    submitted = 2  #: jobs has been submitted to slurm on remote host
    done = 3  #: job finished successfully (all gaussian steps finished) and have been retrieved from host
    failed = 4  #: job failed
    incomplete = 5  #: job is incomplete, it should be resubmitted
    uploaded = 6  #: job has been uploaded to the DB succesfully
    duplicate = 7 #: exactly the same job already exists in database


@dataclass
class slurm_job:
    """Dataclass for slurm job.

    :param can: canonical smiles
    :type can: str
    :param inchi: inchi
    :type inchi: str
    :param inchikey: inchikey
    :type inchikey: str
    :param conformation: conformation number
    :type conformation: int
    :param charges: formal charges for all atoms
    :type charges: list
    :param connectivity_matrix: connectivity matrix
    :type connectivity_matrix: np.ndarray
    :param conformation: conformation number
    :type conformation: int
    :param max_num_conformers: max number of conformers generated for the molecule
    :type max_num_conformers: int
    :param conformer_engine: conformer engine
    :type conformer_engine: str
    :param tasks: gaussian tasks tuple
    :type tasks: tuple
    :param config: gaussian configuration parameters dictionary
    :type config: dict
    :param job_id: job id on the remote host
    :type job_id: int
    :param directory: job local directory
    :type directory: str
    :param base_name: job local base_name
    :type base_name: str
    :param status: slurm_status of the job
    :type status: slurm_status
    :param n_submission: number of times the job has been submitted
    :type n_submission: int
    :param n_success_tasks: number of successfully completed tasks
    :type n_success_tasks: int
    """

    # molecule info
    can: str
    inchi: str
    inchikey: str
    elements: list
    charges: list
    connectivity_matrix: np.ndarray
    conformation: int

    # configuration info
    max_num_conformers: int
    conformer_engine: str
    tasks: tuple
    config: dict

    # slurm variables
    job_id: int
    directory: str
    remote_dir: str
    base_name: str
    status: slurm_status
    n_submissions: int
    n_success_tasks: int
