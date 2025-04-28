# -*- coding: utf-8 -*-
"""Profile the Nested Sampling code with a simple (physical) runcard.
Run using:
    python -m cProfile -s tottime profile_ns.py > <log_file.log>
"""

import pathlib

from smefit.log import setup_console
from smefit.runner import Runner

runcard_path = pathlib.Path(__file__).absolute().parents[1] / "runcards"
fit_card = "test_runcard"

# normal profiling
setup_console(None)
runner = Runner.from_file(runcard_path, fit_card)
runner.ns()


# # profile mpiexec
# os.system("mpiexec -n 2 smefit NS runcards/test_runcard")
