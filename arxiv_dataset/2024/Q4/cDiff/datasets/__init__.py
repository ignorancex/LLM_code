# datasets/__init__.py

# Dictionary to map dataset names to module paths and import functions
DATASET_CONFIG = {
    "cos": {
        "module": "datasets.cos",
        "dataset_generator": "return_cos_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_cos_data",
    },
    "g_and_k": {
        "module": "datasets.g_and_k",
        "dataset_generator": "return_g_and_k_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_g_and_k_data",
    },
    "normal_gamma": {
        "module": "datasets.normal_gamma",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "witch_hat": {
        "module": "datasets.witch_hat",
        "dataset_generator": "return_witch_hat_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_witch_hat_data",
    },
    "stochastic_vol": {
        "module": "datasets.stochastic_vol",
        "dataset_generator": "return_stochastic_vol_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "lotka_volterra": {
        "module": "datasets.lotka_volterra",
        "dataset_generator": "return_lotka_volterra_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "socks": {
        "module": "datasets.socks",
        "dataset_generator": "return_socks_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "species_sampling": {
        "module": "datasets.species_sampling",
        "dataset_generator": "return_species_dl",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "dirichlet_multinomial": {
        "module": "datasets.dirichlet_multinomial",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "dirichlet_laplace": {
        "module": "datasets.dirichlet_laplace",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "possion_gamma": {
        "module": "datasets.possion_gamma",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "markov_switch": {
        "module": "datasets.markov_switch",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "normal_wishart": {
        "module": "datasets.normal_wishart",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "fBM": {
        "module": "datasets.fBM",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    },
    "minnesota": {
        "module": "datasets.minnesota",
        "dataset_generator": "return_dl_ds",
        "sample_theta": "sample_theta",
        "sample_data": "sample_y",
    }
}


def load_dataset(dataset_name):
    """Dynamically load dataset components based on dataset name."""
    if dataset_name not in DATASET_CONFIG:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    config = DATASET_CONFIG[dataset_name]
    module = __import__(config["module"], fromlist=True)

    # Dynamically get the function references
    dataset_generator = getattr(module, config["dataset_generator"])
    sample_theta = getattr(module, config["sample_theta"])
    sample_data = getattr(module, config["sample_data"])

    return dataset_generator, sample_theta, sample_data


if __name__ == "__main__":
    dataset_generator, sample_theta, sample_data = load_dataset("minnesota")
    print(dataset_generator)
    print(sample_theta)
    print(sample_data)