import math


def ln_P_X_equals_k(n, k):
    # Compute ln P(X = k) for binomial distribution B(n, 0.5)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def logsumexp(a):
    # Compute log(sum(exp(a_i))) for a list of log probabilities
    a_max = max(a)
    sum_exp = sum(math.exp(ai - a_max) for ai in a)
    return a_max + math.log(sum_exp)


n_list = [20, 30, 32, 48, 64, 100, 256]

for n in n_list:
    ln_P_list = []
    found_x = None
    for k in range(n // 2 + 1):
        ln_P_k = ln_P_X_equals_k(n, k)
        ln_P_list.append(ln_P_k)
        ln_P_X_less_than_x = logsumexp(ln_P_list)
        P_X_less_than_x = math.exp(ln_P_X_less_than_x)
        if P_X_less_than_x > 5e-5:
            break
        else:
            x_candidate = k + 1  # Since x = k + 1
            found_x = x_candidate
            false_positive_rate = 2 * P_X_less_than_x  # Two-sided test
    if found_x is not None:
        print(
            f"For n = {n}, the largest x is {found_x}, corresponding false positive rate is {false_positive_rate:.5g}")
    else:
        print(f"For n = {n}, no x found within the threshold.")
