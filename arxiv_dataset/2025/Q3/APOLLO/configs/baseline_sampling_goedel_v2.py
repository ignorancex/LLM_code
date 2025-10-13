from prover.utils import AttrDict
from prover.algorithms import Sampling

# verifier
lean_max_concurrent_requests = 8
lean_memory_limit = 10
lean_timeout = 120

# model
batch_size = 4
mode = 'cot_goedel_v2' # chat templates can be changed in prover/utils.py
pass_ = 32
model_path = 'Goedel-LM/Goedel-Prover-V2-8B'

model_args = AttrDict(
    mode=mode,  
    temperature=0.6,
    max_tokens=32768,
    top_p=0.95,
)

# algorithm
n_search_procs = 16
sampler = dict(
    algorithm=Sampling,
    sample_num=pass_,
    log_interval=32,
)