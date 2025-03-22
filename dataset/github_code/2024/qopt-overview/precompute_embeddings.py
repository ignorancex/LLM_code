"""Precomputes (DWave) embeddings for the instances in QUBO folder."""

import pandas as pd

from minorminer import find_embedding
from time import time
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler
from dwave.system import FixedEmbeddingComposite

import sys
sys.path.append("./benchmark")

from bench_qubo import DataBaseQUBO, QUBOSolverDwave
from qubo_utils import get_QUBO_by_ID

from private import DWave_token

import argparse
import json

if __name__ == '__main__':
    main()

def main():
    """Main script code."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        type=str, default="./instances/QUBO")
    parser.add_argument('-e', '--embeddings_directory',
                        type=str, default="./run_logs/dwave/embeddings")
    parser.add_argument('-l', '--list_ids_from',
                        type=str, default=None)
    parser.add_argument('-t', '--threads',
                        type=int, default=1)
    parser.add_argument('-N', '--tries',
                        type=int, default=10)
    parser.add_argument('-T', '--timeout',
                        type=int, default=1000)
    parser.add_argument('-s', '--stats_file',
                        type=str, default="./run_logs/dwave/embeddings/stats.csv")

    args = parser.parse_args()
    directory = args.directory
    embdir = args.embeddings_directory
    stats_file = args.stats_file

    IDs = []
    if args.list_ids_from is not None:
        with open(args.list_ids_from, 'r') as IDs_file:
            IDs = IDs_file.read().split()

        print(f"List of instance IDs supplied:\n {IDs}")

    token = DWave_token

    db = DataBaseQUBO(directory)
    solver = QUBOSolverDwave(db, None)
    # setup sampler
    sampler = DWaveSampler(token=token)
    a = None
    b = None

    stats = pd.DataFrame(columns =[
        "qubo_file", "instance_id", "instance_type", "qubo_vars",
        "emb_time", "emb_success", "emb_file"])

    for instance_id in db.get_instance_ids():
        if (len(IDs)>0) and instance_id not in IDs:
            print(f"{instance_id} is not among the supplied IDs, skipping...")
            continue

        print(f"Building an embedding for {instance_id}...")
        Q, P = solver.get_data(instance_id, a, b)
        bqm = BinaryQuadraticModel(P, Q / 2, vartype = {0, 1})

        # precompute the embedding
        Q_dict, Q_const = bqm.to_qubo()
        __, target_edgelist, target_adjacency = sampler.structure

        t_start = time()
        embedding = find_embedding(Q_dict, target_edgelist, verbose=1,
                                   threads=args.threads,
                                   tries=args.tries,
                                   timeout=args.timeout)  # default is 10 tries
        embedding_time = time() - t_start


        if bool(embedding):
            status = '✅ Precomputed an embedding'
        else:
            status = '❌ Failed to find an embedding'

        print(f"{status} for {instance_id}, in {embedding_time:.1f} sec.", flush=True)

        emb_filename = f"{embdir}/{instance_id}.emb.json"

        with open(emb_filename, 'w') as embfile:
            json.dump(embedding, embfile)

        qubo_filename = get_QUBO_by_ID(instance_id, directory)
        with open(qubo_filename, 'r') as qubof:
            qubojs = json.load(qubof)

        stats.loc[len(stats)] = [qubo_filename, instance_id,
                                 qubojs['description']['instance_type'],
                                 len(qubojs['Q']),
                                 embedding_time, bool(embedding),
                                 emb_filename]

        stats.to_csv(stats_file)
