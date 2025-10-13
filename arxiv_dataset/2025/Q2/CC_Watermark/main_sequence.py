import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_bm(m, k, seed, balanced):
  """
  Generates Bm.
  """
  np.random.seed(seed)
  if balanced:
      # Distribute m items among k labels as evenly as possible
      count_per_label = m // k  # how many times each label appears
      leftover = m % k  # how many labels should get 1 extra

      bm = []
      for label in range(k):
          # If label < leftover, give it one extra
          n_label = count_per_label + (1 if label < leftover else 0)
          bm.extend([label] * n_label)

      # Shuffle in place to randomize order
      np.random.shuffle(bm)
      return np.array(bm)
  bm = np.random.choice(k, size=m)
  return bm


def generate_bm_list(m, k, seed, balanced, seq_len=1):
  """
  Generates Bm.
  """
  np.random.seed(seed)
  bm_list = []
  for _ in range(seq_len):
      if balanced:
          count_per_label = m // k  # how many times each label appears
          leftover = m % k  # how many labels should get 1 extra
          bm = []
          for label in range(k):
              # If label < leftover, give it one extra
              n_label = count_per_label + (1 if label < leftover else 0)
              bm.extend([label] * n_label)
          np.random.shuffle(bm)
          bm = np.array(bm)
      else:
          bm = np.random.choice(k, size=m)
      bm_list.append(bm)
  return bm_list




# Function to sample from a watermarked or non-watermark dist with equal probability
# and compute correct detection of green tokens
def sampling(distribution, tilted_distribution, rg_list, seed):
    #np.random.seed(seed)
    # Randomly sample a bit (0 or 1) w probability 1/2
    bit = np.random.choice([0, 1])
    #print('chosen distribution:', bit)
    # Select distribution to sample from based on bit
    chosen_distribution = distribution if bit == 0 else tilted_distribution

    # Sample 1 element from the chosen distribution
    sampled_element = np.random.choice(len(chosen_distribution), size=1, p=chosen_distribution)

    # Determine whether it is from green list (1 in rg_list)
    green_hit = rg_list[sampled_element].item()

    return green_hit, bit

def one_shot_detection_experiment(distribution, tilted_distribution, rg_list, side_info = None, method = 'ccw'):
    results = []
    FPR_results = []
    FNR_results = []
    watermarked_ct = 0
    notwatermarked_ct = 0
    for _ in range(100):
        green_hit, bit = sampling(distribution, tilted_distribution, rg_list, _)
        # compute correct detection or not
        if method=='rg':
          detection_correctness = int(bit == green_hit)
          FNR = int(bit == 1 and green_hit == 0)
          FPR = int(bit == 0 and green_hit == 1)
        elif method == 'ccw':
          decision = int(green_hit==side_info)
          detection_correctness = int(bit == decision)
          FNR = int(bit == 1 and decision == 0)
          FPR = int(bit == 0 and decision == 1)
        else:
          raise ValueError("Invalid method. Use 'rg' or 'ccw'.")
        FNR_results.append(FNR)
        FPR_results.append(FPR)
        watermarked_ct += int(bit == 1)
        notwatermarked_ct += int(bit == 0)
        #print(detection_correctness)
        results.append(detection_correctness)
    detection_accuracy = sum(results) / len(results)
    ave_FPR = sum(FPR_results) / notwatermarked_ct
    ave_FNR = sum(FNR_results) / watermarked_ct
    assert((watermarked_ct+notwatermarked_ct)==100)
    return detection_accuracy, ave_FPR, ave_FNR, bit

def sequence_level_detection_experiment(distribution, tilted_distribution, bm, s, tau,seq_len):
    """
    built on top of the one-shot.
    Inputs - sequence of BM, sequence of S, sequence of threshold tau
    Set - n_iter, n_seq
    Steps:
    1. generate a prior bit - 1 means wm, 0 means not.
    2. generate a sequence xn using the corresponding distribution
    3. for each element in xn - apply indicator
    4. compare average of indicators to threshold to make a decision.
    """
    results = []
    watermarked_ct = 0
    notwatermarked_ct = 0
    n_iter = 30
    FNRs = []
    FPRs = []
    test_list = []
    for _ in range(n_iter):  # performing sequence-level watermarking for n_iten iterations with the tuple (dists, bm,s) for a given tau.
        bit = np.random.choice([0, 1])  # sample prior
        chosen_dist = tilted_distribution if bit else [distribution]*seq_len  # choose distribution to sample according to prior
        xn = sample_seq(chosen_dist)  # sample a sequence of n xs according to chosen distribution
        xn_partition = check_xn_bm(xn,bm)
        # test = np.mean(xn_partition == s)
        test = np.mean(np.array([xn_partition[i] == s[i] for i in range(len(s))]))
        # print(f'test {test}, bit {bit}')
        test_list.append(test)
        # print(f'test = {test}')
        decision = int(test >= tau)
        detection_correctness = int(bit == decision)
        FNR = int(bit == 1 and decision == 0)
        FPR = int(bit == 0 and decision == 1)
        watermarked_ct += int(bit == 1)
        notwatermarked_ct += int(bit == 0)
        results.append(detection_correctness)
        FNRs.append(FNR)
        FPRs.append(FPR)
    detection_accuracy = sum(results) / len(results)

    av_FNR = sum(FNRs)/len(FNRs)
    av_FPR = sum(FPRs)/len(FPRs)
    if watermarked_ct > 0 and notwatermarked_ct > 0:
        av_FNR = sum(FNRs)/watermarked_ct
        av_FPR = sum(FPRs)/notwatermarked_ct
    return detection_accuracy, av_FNR, av_FPR

def sample_seq(ps):
    x = []
    for p in ps:
        sampled_x = np.random.choice(len(p), size=1, p=p)
        x.append(sampled_x)
    return np.array(x)

def check_xn_bm(xn,bm):
    partition_x = []
    for (i,x) in enumerate(xn):
        partition_x.append(bm[i][x])
    return partition_x

def compute_p_tildeY(Q, B, k):
    """
    Given:
      - Q: array of shape (m,) representing probabilities over m items
      - B: array of shape (m,) with values in {1,..,k} indicating
           which label (from 1..k) each item belongs to
      - k: size of the alphabet {1..k} of Y-tilde
    Returns:
      p: a length-k array where p[i] = P(Ytilde = i+1).
         (Note: i+1 in {1..k}).
    """
    Q = np.asarray(Q, dtype=float)
    B = np.asarray(B, dtype=int)
    p = np.zeros(k, dtype=float)

    # Accumulate Q[r] into p[ B[r] - 1 ] because B is 1-based.
    for r in range(len(Q)):
        p[B[r]] += Q[r]

    return p


def coupling_conditional_for_j(p, j):
    """
    Given:
      - p: array of length k, p[i] = probability that Ytilde = i (0-based),
      - j: an integer in {0,1,..,k-1} indicating the S-value of interest
    Implements the optimal coupling from Prop. 2 (S ~ Uniform(k), Ytilde ~ p)
    and returns a length-k array cond[i] = P(S=j | Ytilde=i).

    All indexing here is 0-based internally, meaning:
      - i in {0,..,k-1} is 'Ytilde = i+1' in 1-based
      - j in {0,..,k-1} is 'S = j+1' in 1-based
    """
    p = np.asarray(p, dtype=float)
    k = len(p)

    # If p is exactly uniform, then S=Ytilde w.p.1 is an optimal coupling
    # => P(S=j | Ytilde=i) = 1 if i==j, else 0.
    if np.allclose(p, 1.0 / k):
        cond = np.zeros(k, dtype=float)
        cond[j] = 1.0
        return cond

    # Total variation distance t
    t = 0.5 * np.sum(np.abs(p - 1.0 / k))

    # A = { i : p_i >= 1/k }, A^c its complement
    A_mask = (p >= 1.0 / k)
    AC_mask = ~A_mask

    # We need, for each i, xi(i,l) for l in {0..k-1} so that we can
    # compute P(S=j | Ytilde=i) = xi(i,j)/sum_l xi(i,l).
    # We'll build them row by row but only store the row-sum and the j-th entry.

    cond = np.zeros(k, dtype=float)  # cond[i] = P(S=j | Ytilde=i)

    for i in range(k):
        # Build xi(i,l) for l=0..k-1
        xi_row = np.zeros(k, dtype=float)
        for l in range(k):
            if i == l:
                # diagonal: min(1/k, p[i])
                xi_row[l] = min(1.0 / k, p[i])
            else:
                # off-diagonal
                if A_mask[i] and AC_mask[l]:
                    # formula: (1/t)*((1/k - p_i)*(p_l - 1/k))
                    xi_row[l] = (1.0 / t) * ((1.0 / k - p[i]) * (p[l] - 1.0 / k))
                else:
                    xi_row[l] = 0.0

        row_sum = np.sum(xi_row)
        if row_sum > 0.0:
            cond[i] = xi_row[j] / row_sum
        else:
            # if row_sum==0, the row is degenerate => set cond[i] = 0
            cond[i] = 0.0

    return cond


# correlated channel method (assume binary side_info and binary rg_list)
def tilt_q_CCW_seq(distribution, s, bm, k,seq_len=1):
    # gen tilde{Y} -> gen coupling -> gen tilted distribution
    q_tilde_list = []
    for i in range(seq_len):
        py_tilde = compute_p_tildeY(distribution, bm[i], k)
        cond_s_y = coupling_conditional_for_j(py_tilde, s[i])
        q_tilde = k * distribution * cond_s_y[bm[i]]
        q_tilde_list.append(q_tilde)
    if seq_len == 1:
        q_tilde_list = q_tilde_list[0]
    return q_tilde_list

def tilt_q_CCW(distribution, s, bm, k):
    # gen tilde{Y} -> gen coupling -> gen tilted distribution
    py_tilde = compute_p_tildeY(distribution, bm, k)
    cond_s_y = coupling_conditional_for_j(py_tilde, s)
    q_tilde = k * distribution * cond_s_y[bm]
    return q_tilde


def experiment_z_channel(orig_dist, k=2, n_trials=100,balanced=False, sequential=False):
    accuracy_list = []
    fpr_list = []
    fnr_list = []
    watermarked_ct = 0
    m = len(orig_dist)
    if sequential:
        tau_list = np.linspace(0.4,0.9,20)
        accuracy_means = []
        accuracy_stds = []
        avg_FNRs = []
        avg_FPRs = []
        fresh_bm = True
        fresh_s = True
        for tau in tau_list:
            fnr_list = []
            fpr_list = []
            for itr in tqdm(range(n_trials)):
                # create seq of s, seq of bm, seq of p
                seed = itr
                seq_len = 50
                if fresh_bm:
                    bm = generate_bm_list(m, k, seed, balanced, seq_len)
                else:
                    bm = [generate_bm(m, k, seed, balanced)] * seq_len
                if fresh_s:
                    side_info = np.random.choice(k, size=seq_len)
                else:
                    side_info = np.ones(shape=(seq_len,),dtype=int)*np.random.choice(k, size=1)

                tilt_q_ccw = tilt_q_CCW_seq(orig_dist, side_info, bm, k, seq_len)  # generate tilted distribution
                detection_accuracy, fnr, fpr = sequence_level_detection_experiment(orig_dist, tilt_q_ccw, bm, side_info, tau, seq_len)
                accuracy_list.append(detection_accuracy)
                fnr_list.append(fnr)
                fpr_list.append(fpr)
            accuracy_means.append(np.mean(accuracy_list))
            accuracy_stds.append(np.std(accuracy_list))
            avg_FNRs.append(np.mean(fnr_list))
            avg_FPRs.append(np.mean(fpr_list))
        return accuracy_means, accuracy_stds, tau_list, avg_FNRs, avg_FPRs
    else:
        for itr in tqdm(range(n_trials)):
            side_info = np.random.choice(k)  # sample s
            seed = itr  # fix seed for iter
            bm = generate_bm(m, k, seed, balanced)  # generate RG list
            tilt_q_ccw = tilt_q_CCW(orig_dist, side_info, bm, k)  # generate tilted distribution

            detection_accuracy, ave_FPR, ave_FNR, bit = one_shot_detection_experiment(orig_dist, tilt_q_ccw, bm, side_info,
                                                                                      method='ccw')
            accuracy_list.append(detection_accuracy)
            fpr_list.append(ave_FPR)
            fnr_list.append(ave_FNR)
            if bit == 1:
                watermarked_ct += 1

        # Calculate mean and standard error for each metric
        means = [np.mean(accuracy_list), np.mean(fpr_list), np.mean(fnr_list)]
        stds = [np.std(accuracy_list, ddof=1),  # Standard Error for Accuracy
                np.std(fpr_list, ddof=1),  # Standard Error for FPR
                np.std(fnr_list, ddof=1)]  # Standard Error for FNR
        print('watermark ratio:', watermarked_ct / 100)
        return means, stds

def expRdVsK(m, orig_dist, max_k):
    k_list = [i for i in range(2, max_k + 1)]

    CCW_list_balanced = []
    CCW_std_list_balanced = []
    for k in k_list:
        # correlated channel result
        means_CCW, sems_CCW = experiment_z_channel(orig_dist, k, n_trials=10, balanced=True)
        detection_CCW = means_CCW[0]
        std_CCW = sems_CCW[0]

        print(f'X alphabet size {m}, S alphabet size {k}, mean detection {detection_CCW} with std {std_CCW}')
        CCW_list_balanced.append(detection_CCW)
        CCW_std_list_balanced.append(std_CCW)

    CCW_list = []
    CCW_std_list = []
    for k in k_list:
        # correlated channel result
        means_CCW, sems_CCW = experiment_z_channel(orig_dist, k, n_trials=1000, balanced=False)
        detection_CCW = means_CCW[0]
        std_CCW = sems_CCW[0]

        print(f'X alphabet size {m}, S alphabet size {k}, mean detection {detection_CCW} with std {std_CCW}')
        CCW_list.append(detection_CCW)
        CCW_std_list.append(std_CCW)
    # plt.figure(figsize=(8, 5))
    # plt.plot(k_list, CCW_list)
    # plt.title(f'CC detection vs. k, m={m}')
    # plt.show()
    #
    # plt.figure(figsize=(8, 5))  # Optional: set figure size
    # plt.bar(k_list, CCW_list, color='blue', alpha=0.7)  # You can change the color
    # # Add labels
    # plt.xlabel("k")
    # plt.ylabel("CC detection")
    # plt.title(f'CC detection vs. k, m={m}')
    # plt.show()

    plt.figure(figsize=(8, 5))
    plt.errorbar(k_list, CCW_list, yerr=CCW_std_list, fmt='o-', capsize=5, label='CC')
    plt.errorbar(k_list, CCW_list_balanced, yerr=CCW_std_list_balanced, fmt='o-', capsize=5, label='CC, balanced')
    plt.plot(k_list, [0.5] * len(k_list), '--*', color='red', label='RG@zero_perception')
    plt.xlabel("k", fontsize=14)
    plt.ylabel("CC detection", fontsize=14)
    plt.title(f'CC detection vs. k, m={m}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def expRdVsNTokens(m,orig_dist,k=2):
    pass


def expROC(m,orig_dist,k=2):
    means_CCW, sems_CCW, tau_list, FNRs, FPRs = experiment_z_channel(orig_dist, k, n_trials=200, balanced=True, sequential=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(tau_list, means_CCW, yerr=sems_CCW, fmt='o-', capsize=5, label='CC')
    plt.title(f'Accuracy vs. tau')
    plt.show()

    plt.figure(figsize=(8, 5))
    # plt.errorbar(FNRs, FPRs, yerr=sems_CCW, fmt='o-', capsize=5, label='CC')
    plt.plot(np.array(FPRs), 1-np.array(FNRs), label='CC')
    plt.title(f'ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    m = 20
    q_unif = False
    lambda_val = 0.5
    if q_unif:
        orig_dist = np.ones(m)
        orig_dist = orig_dist/sum(orig_dist)
    else:
        res = (1-lambda_val)/(m-1)
        orig_dist = np.array([lambda_val] + [res]*(m-1))
    exp = 'ROC'


    if exp == 'rd_vs_k':
        expRdVsK(m,orig_dist,max_k=5)
    elif exp == 'rd_vs_ntokens':
        expRdVsNTokens(m,orig_dist,k=2)
    elif exp == 'ROC':
        expROC(m,orig_dist,k=4)




