import archetypes as arch
import numpy as np
from archetypes.datasets import make_archetypal_dataset

shape = (50, 50)

archetypes_shape = (3, 3)


import torch
from tqdm.auto import tqdm

biaa_values = None
aa_values = None

for i in tqdm(range(20), desc="Experiments"):

    # generate archetypes
    generator = np.random.default_rng(i)
    archetypes = generator.uniform(size=archetypes_shape)

    # generate data
    data, true_labels = make_archetypal_dataset(archetypes, shape, alpha=0.05, noise=0.1)
    data = torch.tensor(data, dtype=torch.float32)

    true_labels_0 = true_labels[0]
    true_labels_1 = true_labels[1]

    # train model using AA in dim 0

    aa_model_0 = None
    best_loss = np.inf

    for i in tqdm(range(3)):
        model = arch.torch.AA(archetypes_shape[0], shape[0], shape[1])
        _ = model.train(data.T, n_epochs=2_000)

        loss = model.losses[-1]

        if loss < best_loss:
            best_loss = loss
            aa_model_0 = model


    # train model using AA in dim 1
    aa_model_1 = None
    best_loss = np.inf

    for i in tqdm(range(3)):
        model = arch.torch.AA(archetypes_shape[1], shape[1], shape[0])
        _ = model.train(data, n_epochs=2_000)

        loss = model.losses[-1]

        if loss < best_loss:
            best_loss = loss
            aa_model_1 = model


    # train model using BiAA

    biaa_model = None
    best_loss = np.inf

    for i in tqdm(range(3)):
        model = arch.torch.BiAA(archetypes_shape, shape[0], shape[1])

        _ = model.train(data.T, n_epochs=2_000)

        loss = model.losses[-1]

        if loss < best_loss:
            best_loss = loss
            biaa_model = model

    aa_betas_0 = aa_model_0.B
    aa_betas_1 = aa_model_1.B.T

    aa_archetypes = (aa_betas_0 @ data @ aa_betas_1).detach().numpy()
    # clip values to [0, 1]
    aa_archetypes = np.clip(aa_archetypes, 0, 1)

    aa_values = np.concatenate((aa_values, aa_archetypes.flatten())) if aa_values is not None else aa_archetypes.flatten()

    biaa_archetypes = biaa_model.Z.detach().numpy()

    # clip values to [0, 1]
    biaa_archetypes = np.clip(biaa_archetypes, 0, 1)

    biaa_values = np.concatenate((biaa_values, biaa_archetypes.flatten())) if biaa_values is not None else biaa_archetypes.flatten()

    # save results
    np.save("aa_values.npy", aa_values)
    np.save("biaa_values.npy", biaa_values)

