from root.sims.plot import plot_with_config, prepare_plot_data


def plot_at_dims_64():
    plot_inputs = prepare_plot_data(
        "lower_dims_correlated",
        [
            ("k_max_lower_dims_n2", "n=2"),
            ("k_max_lower_dims_n6", "n=6"),
            ("k_max_lower_dims_exp", "n=exp"),
        ],
        threshold=100,
    )
    plot_with_config(plot_inputs, "Data-dependent K_max")
