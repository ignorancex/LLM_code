#!/usr/bin/env python3
"""
Benchmark pure activation layers:

  1. MultiVectorAct               (reference, Python)
  2. MultiVectorActOpt5 backend=c (packed-blades C kernel)

Outputs:
    python  :  xx.xx ms
    opt_c   :  xx.xx ms
"""

import time
import torch
from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.gcan import MultiVectorAct
from gcan_opt5 import MultiVectorActOpt5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def ms_per_run(module, x, runs=50, warmup=5):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        t0 = time.time()
        for _ in range(runs):
            module(x)
        t1 = time.time()
    return (t1 - t0) * 1000.0 / runs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # --- problem size ------------------------------------------------------- #
    BATCH      = 256      # samples
    CHANNELS   = 256      # multivector channels
    BLADES     = (0, 1, 2, 3, 4, 5, 6, 7)
    AGG_MODE   = "mean"  # "linear" | "sum" | "mean"
    ALGEBRA    = CliffordAlgebra([1, 1, 1])
    X          = torch.randn(BATCH, CHANNELS, len(BLADES))

    # --- build three modules ------------------------------------------------ #
    act_ref = MultiVectorActOpt5(
        channels      = CHANNELS,
        algebra       = ALGEBRA,
        input_blades  = BLADES,
        kernel_blades = BLADES,
        agg           = AGG_MODE,
        backend       = "py",
    )

    act_opt_c = MultiVectorActOpt5(
        channels      = CHANNELS,
        algebra       = ALGEBRA,
        input_blades  = BLADES,
        kernel_blades = BLADES,
        agg           = AGG_MODE,
        backend       = "c",
    )

    # --- keep weights identical (only needed for "linear") ------------------ #
    if AGG_MODE == "linear":
        for tgt in (act_ref, act_opt_c):
            tgt.conv.load_state_dict(act_ref.conv.state_dict())

    # --- correctness check -------------------------------------------------- #
    with torch.no_grad():
        out_ref    = act_ref(X)
        out_c_opt  = act_opt_c(X)

    torch.testing.assert_close(out_c_opt,  out_ref, atol=1e-5, rtol=1e-5)

    # --- timing ------------------------------------------------------------- #
    for tag, mod in [("python", act_ref),
                     ("opt_c",  act_opt_c)]:
        t = ms_per_run(mod, X)
        print(f"{tag:7s}: {t:8.2f} ms")


if __name__ == "__main__":
    main()
