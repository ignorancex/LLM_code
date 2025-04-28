import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import logging

from open_biomed.data import Molecule, Protein, Text
from open_biomed.models.foundation_models.biomedgpt import BioMedGPT4Chat

def chat_biomedgpt():
    agent = BioMedGPT4Chat.from_pretrained("./checkpoints/biomedgpt-10b/", "cuda:4")
    molecule = Molecule.from_smiles("CC(=CCC1=CC(=CC(=C1O)CC=C(C)C)/C=C/C(=O)C2=C(C=C(C=C2)O)O)C")
    protein = Protein.from_fasta("MAKEDTLEFPGVVKELLPNATFRVELDNGHELIAVMAGKMRKNRIRVLAGDKVQVEMTPYDLSKGRINYRFK")
    agent.append_molecule(molecule)
    print(agent.chat(Text.from_str("Please describe this molecule.")))
    agent.reset()
    agent.append_protein(protein)
    print(agent.chat(Text.from_str("What is the function of this protein?")))

CHAT_UNIT_TESTS = {
    "biomedgpt": chat_biomedgpt,
}

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="biomedgpt")
    args = parser.parse_args()

    if args.model not in CHAT_UNIT_TESTS:
        raise NotImplementedError(f"{args.model} is not currently supported for chat!")
    else:
        CHAT_UNIT_TESTS[args.model]()