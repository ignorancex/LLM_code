import numpy as np

factor = 1.6
config = {
    "d": 10,
    "l1": 0,
    "l2": 1.0,
    "N": 20000,
    "m_ls": np.array([40000, 60000, 80000]),
    "n0": 200,
    #"mul_ls": np.array([1, factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7]),
    "mul_ls": np.array(
        [1, factor, factor ** 2, factor ** 3, factor ** 4, factor ** 5, factor ** 6, factor ** 7, factor ** 8,
         factor ** 9, factor ** 10, factor ** 11, factor ** 12]),
    "M": 500,
    "Ltype": 'const',
    "Vtype": 'random',
}

train_config = {
    "learning_rate": 0.005,
    "decay_rate": 0.95,
    "decay_epoch": 100,
    "epochs": 1000,
}