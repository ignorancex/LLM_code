import numpy as np
import matplotlib.pyplot as plt

# parameters matching your C benchmark
B_set = [32, 64, 128, 256]
C_set = [32, 64, 128, 256]
K_set = [4, 8, 16]
I = 16  # num_blades
REPEAT = 1

modes = {
    0: 'linear',
    1: 'sum',
    2: 'mean',
}

def baseline_pre_sigmoid(v, weights, bias, B, C, NB, K, kernel_indices, agg_mode):
    """
    Python re-implementation of the baseline C multivector_act_forward,
    but returns a flat list of the 'act' values before calling sigmoid.
    """
    pre = []
    for b in range(B):
        for c in range(C):
            act = 0.0
            if agg_mode == 0:
                # linear
                for k in range(K):
                    blade = kernel_indices[k]
                    idx = b*C*NB + c*NB + blade
                    act += v[idx] * weights[c*K + k]
                act += bias[c]
            elif agg_mode == 1:
                # sum
                for k in range(K):
                    blade = kernel_indices[k]
                    idx = b*C*NB + c*NB + blade
                    act += v[idx]
            elif agg_mode == 2:
                # mean
                for k in range(K):
                    blade = kernel_indices[k]
                    idx = b*C*NB + c*NB + blade
                    act += v[idx]
                act /= K
            else:
                act = 0.0
            pre.append(act)
    return pre

def main():
    np.random.seed(0)

    # accumulate all pre-sigmoid values
    all_pre = {mode: [] for mode in modes.values()}

    for mode, mode_name in modes.items():
        print(f"Collecting pre-sigmoid values for mode = {mode_name}")
        for B in B_set:
            for C in C_set:
                for K in K_set:
                    kernel_indices = np.arange(K, dtype=int)
                    for _ in range(REPEAT):
                        # generate random data
                        v = np.random.rand(B*C*I).astype(np.float32)
                        w = np.random.rand(C*K).astype(np.float32) if mode == 0 else None
                        b = np.random.rand(C).astype(np.float32)       if mode == 0 else None

                        pre = baseline_pre_sigmoid(
                            v, w, b,
                            B, C, I, K,
                            kernel_indices, mode
                        )
                        all_pre[mode_name].extend(pre)

    # Build and plot histograms
    bins = np.linspace(-10, 10, 201)  # adjust range as needed
    plt.figure(figsize=(12, 6))
    for i, (mode_name, values) in enumerate(all_pre.items(), 1):
        plt.subplot(1, 3, i)
        plt.hist(values, bins=bins, log=True, color='C'+str(i-1))
        plt.title(f"{mode_name} pre-sigmoid")
        plt.xlabel("act value (pre-sigmoid)")
        plt.ylabel("count (log scale)")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("expf_input_histograms.png")
    print("Histogram saved to expf_input_histograms.png")

if __name__ == "__main__":
    main()
