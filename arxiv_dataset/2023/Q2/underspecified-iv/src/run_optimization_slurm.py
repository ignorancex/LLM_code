"""
Run file for computation of confidence interval for different methods

----
Simulated Data is not saved, but will be immediately used for method computation purposes

"""

import json
from absl import app
from absl import flags
from absl import logging
import os

from run_optimization import run_optimization
from visualization import visualize_results



# ---------------------------- INPUT/OUTPUT -----------------------------------
flags.DEFINE_string("fig_path", "/home/haicu/elisabath.ailer/Projects/UnderspecifiedIV/Output/",
                    "Path to the output directory (for results).")

# ------------------------------ MISC -----------------------------------------
flags.DEFINE_integer("seed", 2022, "The random seed.")


# Scnario Flags
flags.DEFINE_integer("n", 1000, "Number of samples per experiment.")
flags.DEFINE_integer("p", 10, "Number of treatment variables.")
flags.DEFINE_integer("d", 10, "Number of instruments.")
flags.DEFINE_integer("d_id", 10, "Number of instruments that are relevant for the computation.")
flags.DEFINE_integer("n_runs", 500, "Number of random initializations.")
flags.DEFINE_integer("n_rounds", 10, "Optimization: stopping after this many experiments.")
flags.DEFINE_integer("d_max", 4, "Optimization: maximum of instruments in one set.")


FLAGS = flags.FLAGS
# =============================================================================
# MAIN
# =============================================================================

def main(_):

    # ---------------------------------------------------------------------------
    # Directory setup, save flags, set random seed
    # ---------------------------------------------------------------------------
    FLAGS.alsologtostderr = True

    d = FLAGS.d
    p = FLAGS.p
    n = FLAGS.n
    d_max = FLAGS.d_max
    n_rounds = FLAGS.n_rounds
    fig_path = FLAGS.fig_path
    n_runs = FLAGS.n_runs
    seed = FLAGS.seed
    d_id = FLAGS.d_id


    # ----------------------------------------------------------------------------------------------------------------
    # 0. Initialization
    # ----------------------------------------------------------------------------------------------------------------
    name_id = "p" + str(p) + "_d" + str(d) + "_d_id" + str(d_id) + "_d_max" + str(d_max) + "_n_rounds" + str(
        n_rounds)

    fig_save_path = os.path.join(fig_path, str(name_id))

    try:
        os.makedirs(fig_save_path)
    except FileExistsError:
        pass

    logging.info(f"Save all output to {fig_save_path}...")

    FLAGS.log_dir = fig_save_path
    logging.get_absl_handler().use_absl_log_file(program_name="run")

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(fig_save_path, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")

    # ----------------------------------------------------------------------------------------------------------------
    # 1. Run Optimization
    # ----------------------------------------------------------------------------------------------------------------
    run_optimization(seed, n_runs, n, p, d, d_id, d_max, n_rounds, name_id, fig_save_path)

    # ----------------------------------------------------------------------------------------------------------------
    # 2. Visualize Results
    # ----------------------------------------------------------------------------------------------------------------
    visualize_results(name_id, fig_save_path)

    logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)







































