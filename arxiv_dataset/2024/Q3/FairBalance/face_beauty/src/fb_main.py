from fb_exp import exp
import pandas as pd
import numpy as np

def run(base="Average", repeats = 10):
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant"]
    runner = exp(rating_cols = [base])
    result = None
    for _ in range(repeats):
        test_result = runner.run(base=base, treatments=treatments)
        if result is None:
            result = {key: test_result[key] if key == "Treatment" else [[value] for value in test_result[key]] for key in test_result}
            continue
        for key in test_result:
            if key == "Treatment":
                continue
            for i, value in enumerate(test_result[key]):
                result[key][i].append(value)
    for key in result:
        if key == "Treatment":
            continue
        result[key] = [
            "%.2f (%.2f)" % (np.median(l), np.quantile(l, 0.75) - np.quantile(l, 0.25)) for l in
            result[key]]
    df_test = pd.DataFrame(result)
    df_test.to_csv("../results/fb_" + base + ".csv", index=False)


if __name__ == "__main__":
    run(base="Average", repeats = 10)