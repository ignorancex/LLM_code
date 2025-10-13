from prover.utils import AttrDict
from prover.algorithms import Sampling

# verifier
lean_max_concurrent_requests = 4
lean_memory_limit = 10
lean_timeout = 120

# model
batch_size = 4
mode = 'cot' # chat templates can be changed in prover/utils.py
pass_ = 32
model_path = 'Goedel-LM/Goedel-Prover-SFT'

model_args = AttrDict(
    mode=mode,  
    temperature=1,
    max_tokens=2048,
    top_p=0.95,
)

# algorithm
n_search_procs = 16
sampler = dict(
    algorithm=Sampling,
    sample_num=pass_,
    log_interval=32,
)