import math
import scipy.stats as st

# here pos is the number of successes
# n is the total number of samples
# and confidence is in [0, 1], usually .95
def ci(pos, n, confidence):
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    lower = (phat + z * z / (2 * n) - z *
             math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) /
            (1 + z * z / n)
    upper = (phat + z * z / (2 * n) + z *
             math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) /
            (1 + z * z / n)
    return (lower, upper)
