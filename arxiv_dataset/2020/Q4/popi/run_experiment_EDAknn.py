from popt import run


if __name__ == "__main__":
    num_exps = 3

    model_names = ['mutation1']

    exp_signature = '_exp_'

    methods = ['EDAknn']

    # optimization hypeparams
    generations = 1000
    pop_size = 100

    patience = 50

    # mutation & recombination
    std = 0.1
    gamma = 0.5
    CR = 0.9

    # other optimizer hyperparams
    p = -1.
    best = False

    # possible values
    low = [0.00 for i in range(18)]
    low[2] = 550.
    low[3] = 350.
    low[5] = 300.
    low[6] = 76400.
    low[7] = 57800.
    low[8] = 20.
    low[9] = 80.
    low[11] = 2000.
    low[12] = 20.
    low[13] = 80.

    high = [10. for i in range(18)]
    high[2] = 600.
    high[3] = 400.
    high[5] = 350.
    high[6] = 76450.
    high[7] = 57850.
    high[8] = 50.
    high[9] = 100.
    high[11] = 2050.
    high[12] = 50.
    high[13] = 100.


    # simulation values
    sim_start = 0.0
    sim_end = 1.0
    sim_points = 30

    noise = 0.01

    # objective
    a = -100.
    b = 100.

    dist_name = 'norm'
    scale = 1.

    # observable data
    indices = [0, 1, 2, 5, 8]

    # directory
    dir_model = '/popi/'

    for model_name in model_names:
        for m in methods:
            for i in range(num_exps):
                run(exp_sign=model_name[:-1] + exp_signature + str(i + 1) + '_', method_name=m, mod_name=model_name,
                    sim_start=sim_start, sim_end=sim_end, sim_points=sim_points,
                    generations=generations, pop_size=pop_size, clip_min=low, clip_max=high, a=a, b=b,
                    scale=scale, p=p, std=std, gamma=gamma, CR=CR, best=best, dist_name=dist_name, low=low,
                    high=high,
                    indices=indices,
                    compartment=True,
                    patience=patience,
                    noise=noise,
                    dir_model=dir_model,
                    slash='/')
