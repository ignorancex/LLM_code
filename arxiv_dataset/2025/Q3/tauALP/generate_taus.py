import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def run_simulation(
    num_events, base_filename, beam_energy, soft_or_hard, pTcut, seed, beam_type
):
    # Execute the C++ program with the base filename and the number of events as command line arguments
    command = [
        "./ntaus",
        str(num_events),
        base_filename,
        str(beam_energy),
        str(soft_or_hard),
        str(pTcut),
        str(seed),
        str(beam_type),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the result status
    if result.returncode != 0:
        print(f"Error with {base_filename}: {result.stderr.decode()}")
    else:
        print(f"Completed: {base_filename} with {num_events} events")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tau simulations.")
    parser.add_argument(
        "--beam_energy",
        type=float,
        required=True,
        help="Beam energy for the simulation",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="name for this simulation to be used in saving files, e.g., NuMI_120GeV",
    )
    parser.add_argument(
        "--soft_or_hard",
        type=str,
        required=True,
        help="whether to simulate with soft or hard QCD events. Options are 'soft', 'hard', or 'both'.",
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        required=True,
        help="whether to simulate a fixed target or collider setup. Options are 'fixed' or 'collider'.",
    )
    parser.add_argument(
        "--pTcut",
        type=str,
        required=False,
        default="1.0",
        help="the pThat cut to be used when avoiding double counting in QCDsoft vs QCDhard events. Should be O(1.0 GeV).",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=False,
        default="666",
        help="integer seed for Pythia. Each event file will have seed = (seed+i) as an input, where i is the generator index (0 to n_cores).",
    )
    parser.add_argument(
        "--n_events",
        type=str,
        required=False,
        default=1000,
        help="Number of tau events to generate with Pythia for each job.",
    )
    args = parser.parse_args()
    # Configuration for different runs
    tasks = [
        (
            str(i),
            str(i + int(args.seed)),
        )
        for i in range(8)
    ]

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run_simulation,
                args.n_events,  # number of events
                args.name + "_" + task[0],  # file names (0 - n_cores)
                str(args.beam_energy),
                args.soft_or_hard,
                args.pTcut,
                task[1],  # seed
                args.beam_type,
            )
            for task in tasks
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                # Result will be None as the task doesn't return anything
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")


if __name__ == "__main__":
    main()
