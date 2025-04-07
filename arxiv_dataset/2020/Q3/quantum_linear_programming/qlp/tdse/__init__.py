"""Library to solve the time dependent schrodinger equation in the many-body Fock space.
"""

from qlp.tdse.schedule import AnnealSchedule
from qlp.tdse.example import embed_qubo_example
from qlp.tdse.computation import TDSE, convert_params, add_jchaos