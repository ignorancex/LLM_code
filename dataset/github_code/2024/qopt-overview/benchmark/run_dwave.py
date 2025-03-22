"""An example wrapper script for running experiments on D-Wave machine.

Relies upon :py:mod:`bench_qubo.DataBaseQUBO` to store instances and
:py:class:`bench_qubo.QUBOSolverDwave` to handle interactions with the (remote)
hardware.
"""
import datetime
import argparse
import os.path

from bench import __version__ as version
from bench_qubo import DataBaseQUBO, QUBOSolverDwave
from private import DWave_token

def main():
    """Main script code."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str)
    parser.add_argument('--ids', nargs='+', type=str)
    parser.add_argument('--directory', type=str, default=r'..\instances\QUBO')
    parser.add_argument('--directory-embeddings', type=str, default=r'..\run_logs\dwave\embeddings')

    args = parser.parse_args()

    benchmark_name = args.name
    allowed_instance_ids = args.ids
    directory = args.directory
    directory_embeddings = args.directory_embeddings

    def filter_fun(data):
        return data['description']['instance_id'] in allowed_instance_ids
        
    def sort_fun(instance_id, data):
        return (len(data['P']), instance_id)

    db = DataBaseQUBO(directory, filter_fun=filter_fun, sort_fun=sort_fun, directory_embeddings=directory_embeddings)

    token = DWave_token

    solver = QUBOSolverDwave(db, token)

    num_reads_list = [1000]
    a = None
    b = None
    timestamp = datetime.datetime.now().timestamp()

    if not os.path.exists(f'result/{benchmark_name}'):
        os.mkdir(f'result/{benchmark_name}')
    save_file_path = f'result/{benchmark_name}/{benchmark_name}_dwave_{version}_{timestamp}'

    num_repetitions = 1

    run_list = [
        (instance_id, (num_reads, a, b))
        for num_reads in num_reads_list for _ in range(num_repetitions) for instance_id in db.get_instance_ids()]

    print(f'{solver.name}: {list(db.get_instance_ids())} -> {save_file_path}')
    solver.benchmark(run_list, save_file_path)


if __name__ == '__main__':
    main()
