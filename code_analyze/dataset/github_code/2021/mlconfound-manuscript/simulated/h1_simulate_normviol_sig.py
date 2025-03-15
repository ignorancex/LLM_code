
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlconfound.stats import full_confound_test, partial_confound_test
from mlconfound.simulate import simulate_y_c_yhat, polynomial, sigmoid


def run(e, d, w_yc, wyyhat, w_cyhat, Ns):
    def workhorse(_random_state):
        # simulate
        ps = []
        for n in Ns:
            y, c, yhat = simulate_y_c_yhat(w_yc=w_yc,
                                           w_yyhat=w_yyhat, w_cyhat=w_cyhat,
                                           n=n,
                                           random_state=_random_state,
                                           delta=d,
                                           epsilon=e,
                                           nonlin_trf_fun=sigmoid)

            res_gam = partial_confound_test(y, yhat, c,
                                            cat_y=False,
                                            cat_yhat=False,
                                            cat_c=False,
                                            num_perms=1000,
                                            cond_dist_method='gam',
                                            random_state=_random_state,
                                            n_jobs=1,
                                            progress=False)

            ps.append(res_gam.p)
        return np.array(ps)

    rng = np.random.default_rng(42)

    random_sates = rng.integers(np.iinfo(np.int32).max, size=100)
    p50, p100, p500, p1000 = zip(*
                                 Parallel(n_jobs=-1)(
                                     delayed(
                                         workhorse)(
                                         rs) for rs in
                                     random_sates)
                                 )
    ret = [np.sum(np.array(p50) < 0.05)/100,
           np.sum(np.array(p100) < 0.05)/100,
           np.sum(np.array(p500) < 0.05)/100,
           np.sum(np.array(p1000) < 0.05)/100]
    print(ret)
    return ret


###############################################################

res = np.zeros((5, 4))  # dist x Ns

# d=0.1 e=2: wyc=0.1, wyyhat=0.1,  wcyhat=0.06 N=50,100,500,1000

d = 0.1
e = 2
w_yc = 0.1
w_yyhat = 0.1
w_cyhat = 0.06
Ns = [50, 100, 500, 1000]

res[0, :] = run(e, d, w_yc, w_yyhat, w_cyhat, Ns)

# d=1, e=0: wyc=0.33, wyyhat=0.33, wcyhat=0.2, N=50,100,500,1000

d = 1
e = 0
w_yc = 0.33
w_yyhat = 0.33
w_cyhat = 0.2
Ns = [50, 100, 500, 1000]

res[1, :] = run(e, d, w_yc, w_yyhat, w_cyhat, Ns)

# d=1.05, e=-3, wyc=4, wyyhat=4,    wcyhat=2.5,  N=50,100,500,1000

d = 1.05
e = -3
w_yc = 4
w_yyhat = 4
w_cyhat = 2.5
Ns = [50, 100, 500, 1000]

res[2, :] = run(e, d, w_yc, w_yyhat, w_cyhat, Ns)

# d=1.5,  e=-5: wyc=55, wyyhat=55,  wcyhat=40,  N=50,100,500,1000

d = 1.5
e = -5
w_yc = 55
w_yyhat = 55
w_cyhat = 40
Ns = [50, 100, 500, 1000]

res[3, :] = run(e, d, w_yc, w_yyhat, w_cyhat, Ns)


# d=5,   e=-10: wyc=3000000, wyyhat=3000000, wcyhat=2000000, N=50,100,500,1000

d = 5
e = -10
w_yc = 3000000
w_yyhat = 3000000
w_cyhat = 1000000
Ns = [50, 100, 500, 1000]

res[4, :] = run(e, d, w_yc, w_yyhat, w_cyhat, Ns)

results = pd.DataFrame(res, columns=['n=50', 'n=100', 'n=500', 'n=1000'], index=['e=0.1,d=2',
                                                                                 'e=1,d=0',
                                                                                 'e=1.05,d=-3',
                                                                                 'e=1.5,d=-5',
                                                                                 'e=5,d=-10'])
results.to_csv('simulated/results/h1_simulate_normviol_sig.csv')
