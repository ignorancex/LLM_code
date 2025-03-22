import pandas as pd

if __name__ == "__main__":
    results = []

    df = pd.read_csv("data/predict/attack/Large_a3/baseline", delimiter="\t")[:499]
    clean = (df["correct"]).mean()

    # pgd attack
    line = ['pgd']
    line.append(clean)
    for noise in ['0.25', '0.50', '0.75', '1.0', '1.25', '1.5']:
        df = pd.read_csv(f"data/predict/attack/Large_a3/N_{noise}_base", delimiter="\t")[:499]
        frac_correct_accurate = (df["correct"]).mean()
        line.append(frac_correct_accurate)

    results.append(tuple(line))

    # smooth attack
    for m in [1, 4, 8, 16]:
        line = [f'm={m}']
        line.append(clean)
        for noise in ['0.25', '0.50', '0.75', '1.0', '1.25', '1.5']:
            df = pd.read_csv(f"data/predict/attack/Large_a3/N_{noise}_m{m}", delimiter="\t")[:499]
            frac_correct_accurate = (df["correct"]).mean()
            line.append(frac_correct_accurate)

        results.append(tuple(line))

    df = pd.DataFrame.from_records(results, "attack",
                                   columns=["attack", "clean", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5"])
    print(df.to_latex(float_format=lambda f: "{:.2f}".format(f)))

    results = []

    df = pd.read_csv("data/predict/attack/Small_a3/baseline", delimiter="\t")[:499]
    clean = (df["correct"]).mean()

    # pgd attack
    line = ['pgd']
    line.append(clean)
    for noise in ['0.25', '0.50', '0.75', '1.0', '1.25', '1.5']:
        df = pd.read_csv(f"data/predict/attack/Small_a3/N_{noise}_base", delimiter="\t")[:499]
        frac_correct_accurate = (df["correct"]).mean()
        line.append(frac_correct_accurate)

    results.append(tuple(line))

    # smooth attack
    for m in [1, 4, 8, 16]:
        line = [f'm={m}']
        line.append(clean)
        for noise in ['0.25', '0.50', '0.75', '1.0', '1.25', '1.5']:
            df = pd.read_csv(f"data/predict/attack/Small_a3/N_{noise}_m{m}", delimiter="\t")[:499]
            frac_correct_accurate = (df["correct"]).mean()
            line.append(frac_correct_accurate)

        results.append(tuple(line))

    df = pd.DataFrame.from_records(results, "attack",
                                   columns=["attack", "clean", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5"])
    print(df.to_latex(float_format=lambda f: "{:.2f}".format(f)))
