import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from root.data.data import Data
from root.sims.plot import create_x

data = {
    2: "k_max_lower_dims_n2",
    6: "k_max_lower_dims_n6",
    1000: "k_max_lower_dims_exp",
}


# Load and prepare data
def load_data(label):
    y = Data.load(label)
    x = create_x(y, Data.load("lower_dims_correlated"))

    # Remove consecutive values of y that are 500 and larger
    threshold = 500
    new_x, new_y = [], []
    above_threshold = False

    for xi, yi in zip(x, y):
        if yi >= threshold:
            if not above_threshold:
                new_x.append(xi)
                new_y.append(yi)
            above_threshold = True
        else:
            new_x.append(xi)
            new_y.append(yi)
            above_threshold = False

    x, y = new_x, new_y

    return np.array(x), np.array(y)


# Exponential function
def exp_model(x, a, b):
    return a * np.exp(b * x)


# Curve fitting and plotting
plt.figure(figsize=(8, 6))
colors = {2: "blue", 6: "green", 1000: "red"}

for n, label in data.items():
    x, y = load_data(label)
    try:
        popt, _ = curve_fit(exp_model, x, y, p0=[1, 0.1], maxfev=5000)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = exp_model(x_fit, *popt)
        plt.scatter(x, y, color=colors[n], label=f"n={n} (data)")
        plt.plot(x_fit, y_fit, color=colors[n], linestyle="--", label=f"n={n} (fit)")
    except Exception as e:
        print(f"Curve fitting failed for n={n}: {e}")

plt.title("Data-dependent $K_{max}$ vs Mean Hamming Distance")
plt.xlabel("Mean Hamming Distance (d)")
plt.ylabel("$K_{max}$")
plt.legend()
plt.grid()
plt.show()
