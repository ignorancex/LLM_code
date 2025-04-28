from typing import List, Optional, Union

import contextlib
import numpy as np
import os
import sys
import pytorch_lightning as pl

from open_biomed.core.tool import Tool
from open_biomed.data import Molecule, Protein
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class PocketMoleculeDocking(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Ligand-pocket docking.',
            'Inputs: {"molecule": the ligand, "pocket": the pocket}',
            "Outputs: A new molecule with 3D coordinates indicating the binding pose."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("pocket_molecule_docking", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("pocket_molecule_docking", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return DockingEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/rmsd",
            output_str="-{val_rmsd:.4f}",
            mode="min",
        )

class DockingEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    # TODO: implement RMSD evaluation

class VinaDockTask(Tool):
    def __init__(self, mode: str="dock") -> None:
        self.mode = mode
        
        python_path = sys.executable
        conda_env_root = os.path.dirname(os.path.dirname(python_path))
        self.pdb2pqr_path = os.path.join(conda_env_root, 'bin', 'pdb2pqr30')

    def print_usage(self) -> str:
        return "\n".join([
            'Ligand-receptor docking.',
            'Inputs: {"molecule": the ligand, "protein": the receptor}',
            "Outputs: A float number indicating the AutoDockVina score of the binding."
        ])

    def run(self, molecule: Molecule, protein: Protein) -> Union[List[float], List[str]]:
        sdf_file = molecule.save_sdf()
        pdb_file = protein.save_pdb()
        pos = np.array(molecule.conformer)
        center = (pos.max(0) + pos.min(0)) / 2
        size = pos.max(0) - pos.min(0) + 5
        try:
            from openbabel import pybel
            from meeko import MoleculePreparation
            import subprocess
            from vina import Vina
            import AutoDockTools

            ob_mol = next(pybel.readfile("sdf", sdf_file))
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    preparator = MoleculePreparation()
                    preparator.prepare(ob_mol.OBMol)
                    lig_pdbqt = sdf_file.replace(".sdf", ".pdbqt")
                    preparator.write_pdbqt_file(lig_pdbqt)
            
            prot_pqr = pdb_file.replace(".pdb", ".pqr")
            if not os.path.exists(prot_pqr):
                subprocess.Popen([self.pdb2pqr_path,'--ff=AMBER', pdb_file, prot_pqr],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
            prot_pdbqt = pdb_file.replace(".pdb", ".pdbqt")
            if not os.path.exists(prot_pdbqt):
                prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
                subprocess.Popen(['python3', prepare_receptor, '-r', prot_pqr, '-o', prot_pdbqt],
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
            
            v = Vina(sf_name='vina', seed=0, verbosity=0)
            v.set_receptor(prot_pdbqt)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=center, box_size=size)
            if self.mode == "minimize":
                score = v.optimize()[0]
                pose_file = f"./tmp/{molecule.name}_{protein.name}_pose"
                with open(pose_file, "w") as f:
                    v.write_pose(pose_file, overwrite=True)
            elif self.mode == 'dock':
                v.dock(exhaustiveness=8, n_poses=1)
                score = v.energies(n_poses=1)[0][0]
                pose_file = "None"
            return [score], [pose_file]
        except ImportError:
            print("AutoDockVina not installed. This function return 0.0.")
            return [0.0], ["0.0"]