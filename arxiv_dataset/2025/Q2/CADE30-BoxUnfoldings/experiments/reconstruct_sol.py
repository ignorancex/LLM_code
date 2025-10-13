##Recomputes solution from cut edges
import os
import subprocess


def run_solver(command, cnf):
    """Runs a command and measures its execution time."""
    try:
        logs = subprocess.run(command, input=cnf, shell=True, check=True, stdout=subprocess.PIPE, text=True).stdout
    except subprocess.CalledProcessError as e:
        ##I have no idea why gimsatul triggers this
        logs = e.stdout

    return logs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Recomputes solution using')
    parser.add_argument('-s', '--solver', type=str, help='path to sat solver', required=True)
    parser.add_argument('-e', '--edges', type=str, help='path to edges file', required=True)
    parser.add_argument('-i', '--input', type=str, help='path to input cnf file', required=True)
    parser.add_argument('-l', '--lines', type=int, nargs='*', help='specific lines to run the solver on', default = [])
    parser.add_argument('-o', '--output', type=str, help='path to output files', default = 'sols/')

    args = parser.parse_args()
    solver_path = args.solver
    edges_path = args.edges
    lines_to_run = args.lines
    cnf_path = args.input
    output_path = args.output


    os.makedirs(output_path, exist_ok=True)

    vars = 0
    clauses = 0
    with open(cnf_path, 'r') as f:
        lines = f.readlines()
        first_line = lines[0].split()
        base_cnf = "".join(lines[1:])
        vars = int(first_line[2])
        clauses = int(first_line[3])

    with open(edges_path) as f:
        edges = f.readlines()

    if len(lines_to_run) == 0:
        lines_to_run = [x for x in range(len(edges))]

    for i in lines_to_run:
        cur_edges = edges[i].split()[1:-1]
        cur_cnf = f"p cnf {vars} {clauses + len(cur_edges)}\n" + base_cnf + " 0\n".join(cur_edges) + " 0\n"
        solve_command = f"{solver_path} -q"
        sol = run_solver(solve_command, cur_cnf)

        with open(output_path + f"/solution_{i}.txt", 'w') as f:
            f.write(sol)
