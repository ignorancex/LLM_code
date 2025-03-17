import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

phi = (1 + 5**0.5) / 2

sns.set_context("paper")
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams.update(
    {
        "figure.figsize": [5 * phi, 5],
        "figure.dpi": 300,
        "font.size": 32,
    }
)
