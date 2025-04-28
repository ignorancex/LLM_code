import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error

multiplier = 1

dir_data = "../data/ml"

substring = "sps_cumsum_norm"

experimental_contexts = [None,
                         "1p2uW_3000cps",
                         "2p5uW_4000cps",
                         "4uW_4100cps",
                         "8uW_5100cps",
                         "10uW_6000cps",
                         "10uW_12000cps",
                         "20uW_7000cps",
                         "30uW_7000cps",
                         None]

fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex = True, sharey = True)
len_max = 0
idx = 0

for experimental_context in experimental_contexts:
    if not experimental_context is None:
        df_data = pd.read_csv("%s/%s_%s.csv" % (dir_data, substring, experimental_context))
        g_best = df_data["g_best"]
        g_fit = df_data["g_fit"]
        len_max = max(len_max, len(g_fit))

        column_names = df_data.columns
        X = df_data.drop(columns = column_names[-7:])
        y = g_best

        scaler = StandardScaler()
        regressor = SGDRegressor(eta0 = 0.0003 * multiplier, alpha = 1.0e-7)

        y_predicted = pd.Series([np.nan] * len(y))

        for index, row in X.iterrows():
            X_instance = row.values.reshape(1, -1)
            y_instance = y[index]

            # Predict.
            if index == 0:
                pass
            else:
                X_instance_scaled = scaler.transform(X_instance)
                y_instance_predicted = regressor.predict(X_instance_scaled)
                y_predicted[index] = y_instance_predicted

            # Fit.
            scaler.partial_fit(X_instance, [y_instance])
            X_instance_scaled = scaler.transform(X_instance)
            regressor.partial_fit(X_instance_scaled, [y_instance])

        ax = axes[int(idx/2), idx%2]

        events_per_sec = df_data["events"].iloc[-1]/(len(y)*10)

        ax.plot([1000/events_per_sec, 1000/events_per_sec], [0, 1], color="black", linestyle=":")
        ax.plot([10000/events_per_sec, 10000/events_per_sec], [0, 1], color="black", linestyle=":")

        x = range(10, (1 + len(y))*10, 10)

        if idx == 0:
            ax.plot(x, g_best, label = "Best", linestyle="--")
            ax.plot(x, g_fit, label = "Fit")
            ax.plot(x, y_predicted, label = "SGD")
        else:
            ax.plot(x, g_best, linestyle="--")
            ax.plot(x, g_fit)
            ax.plot(x, y_predicted)

        ax.set_title(experimental_context)
        ax.grid(alpha=0.2)

        idx += 1

ax.set_xlim(10, (1 + len_max)*10)
ax.set_ylim(0, 1)
ax.set_xscale("log")

fig.supxlabel("Measurement (s)")
fig.supylabel("$g^{(2)}(0)$")

# fig.text(0.5, 0.04, "Measurement (s)", ha = "center")
# fig.text(0.04, 0.5, "$g^{(2)}(0)$", va = "center", rotation = "vertical")

fig.legend(bbox_to_anchor=(0.975, 0.96))
# fig.legend(bbox_to_anchor=(0.975, 0.725))

plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()

# Show the plot.
plt.show()