import time
from l2_regression import *
from coreset import *
from generate import *
from optimization import *

#############################################
# evaluate the empirical error for coreset_glse and uniform_1
def evaluate_glse(panel,q,lam,times):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    BETA = []
    RHO = []
    for i in range(times):
        BETA.append(generate_beta(d - 1))
        RHO.append(generate_rho(q,lam))

    evaluate_glse = []
    evaluate_uniform1 = []
    # evaluate_uniform2 = []
    size_glse = []
    construction_time_glse = [0 for i in range(len(eps))]
    opt_time_glse = [0 for i in range(len(eps))]

    # opt_uniform1 = []
    # opt_uniform2 = []

    # compute the optimal regression value on the full dataset
    start = time.time()
    beta, rho, value = IRLS_glse(panel, q)
    print('opt:')
    opt_time = [time.time()-start for i in range(len(eps))]
    print('value:', value, 'time:', time.time() - start)

    for e in range(len(eps)):
        print(eps[e])
        nt_sample = int(2 * q * (d-1) * (d-1) / eps[e] / eps[e] / eps[e])
        start = time.time()
        c_glse = coreset_glse(sen_glse(panel, q, lam), nt_sample)
        construction_time_glse[e] = time.time()-start
        size = len(c_glse)
        size_glse.append(size)
        u1_glse = uniform_1(N*T, size)
        # sample = int(math.sqrt(nt_sample))
        # u2_glse = coreset_nt_and_divide(T,uniform_2(N,T,sample,sample),1)

        # compute the optimal regression value on the coreset
        start = time.time()
        beta, rho, value = IRLS_glse_coreset(panel,c_glse,q)
        print('opt_c:')
        opt_time_glse[e] = time.time() - start + construction_time_glse[e]
        print('value:', value, 'time:', time.time() - start + construction_time_glse[e])

        # compute the optimal regression value on Uni1
        start = time.time()
        beta, rho, value = IRLS_glse_coreset(panel,u1_glse,q)
        print('opt_u1:')
        print('value:', value, 'time:', time.time() - start)

        temp_evaluate_glse = []
        temp_evaluate_uniform1 = []
        for i in range(times):
            beta = BETA[i]
            rho = RHO[i]

            all = glse_obj(panel,beta,rho)
            glse = glse_coreset_obj(panel,beta,rho,c_glse)
            uniform1 = glse_coreset_obj(panel,beta,rho,u1_glse)
            # uniform2 = glse_coreset_obj(panel,beta,rho,u2_glse)
            error_glse = abs(glse-all)/all
            error_uniform1 = abs(uniform1-all)/all
            # error_uniform2 = abs(uniform2 - all) / all

            temp_evaluate_glse.append(error_glse)
            temp_evaluate_uniform1.append(error_uniform1)
            # temp_evaluate_uniform2.append(error_uniform2)

        evaluate_glse.append(temp_evaluate_glse)
        evaluate_uniform1.append(temp_evaluate_uniform1)
        # evaluate_glse.append([np.max(temp_evaluate_glse), np.mean(temp_evaluate_glse), np.std(temp_evaluate_glse), \
        #                      np.sqrt(np.mean(np.square(temp_evaluate_glse)))])
        # evaluate_uniform1.append(
        #    [np.max(temp_evaluate_uniform1), np.mean(temp_evaluate_uniform1), np.std(temp_evaluate_uniform1), \
        #     np.sqrt(np.mean(np.square(temp_evaluate_uniform1)))])
        # evaluate_uniform2.append([np.max(temp_evaluate_uniform2), np.mean(temp_evaluate_uniform2), np.std(temp_evaluate_uniform2)])


    return eps, evaluate_glse, evaluate_uniform1, size_glse, construction_time_glse, opt_time_glse, opt_time


#############################################
# evaluate the empirical error for coreset_glsek and uniform_2
def evaluate_glsek(panel,k,q,lam,times):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    eps = [0.1, 0.2, 0.3, 0.4, 0.5]

    BETA = []
    RHO = []
    for i in range(times):
        BETA.append([])
        RHO.append([])
        for l in range(k):
            BETA[i].append(generate_beta(d - 1))
            RHO[i].append(generate_rho(q,lam))

    evaluate_glsek = []
    evaluate_uniform1 = []
    # evaluate_uniform2 = []
    size_glsek = []
    construction_time_glsek = [0 for i in range(len(eps))]
    opt_time_glsek = [0 for i in range(len(eps))]

    # opt_uniform1 = []
    # opt_uniform2 = []

    # compute the optimal regression value on the full dataset
    start = time.time()
    beta, rho, value = IRLS_glsek(panel, q, k)
    print('opt:')
    opt_time = [time.time() - start for i in range(len(eps))]
    print('value:', value, 'time:', time.time() - start)

    for e in range(len(eps)):
        print(eps[e])
        # coreset size for synthetic dataseet
        n_sample = int(q * k * (d-1) / eps[e])
        t_sample = int(q * (d - 1) / eps[e])
        # coreset size for real-world dataset
        # n_sample = int(q * k * (d - 1) / eps[e])
        # t_sample = int(q * (d - 1) / eps[e])
        start = time.time()
        c_glsek = coreset_glsek(panel, q, lam, n_sample, t_sample)
        construction_time_glsek[e] = time.time()-start
        size = 0
        for s in range(len(c_glsek)):
            size += len(c_glsek[s])-1
        size_glsek.append(size)
        u1_glsek = coreset_nt_and_divide(T,uniform_1(N*T, size),0)
        # u2_glsek = uniform_2(N, T, n_sample, t_sample)

        # compute the optimal regression value on the coreset
        start = time.time()
        beta, rho, value = IRLS_glsek_coreset(panel, c_glsek, q, k)
        print('opt_c:')
        opt_time_glsek[e] = time.time() - start + construction_time_glsek[e]
        print('value:', value, 'time:', time.time() - start + construction_time_glsek[e])

        temp_evaluate_glsek = []
        temp_evaluate_uniform1 = []
        for i in range(times):
            beta = BETA[i]
            rho = RHO[i]
            all = glsek_obj(panel,beta,rho)
            glsek = glsek_coreset_obj(panel,beta,rho,c_glsek)
            uniform1 = glsek_coreset_obj(panel,beta,rho,u1_glsek)
            error_glsek = abs(glsek-all)/all
            error_uniform1 = abs(uniform1-all)/all

            temp_evaluate_glsek.append(error_glsek)
            temp_evaluate_uniform1.append(error_uniform1)

        evaluate_glsek.append([np.max(temp_evaluate_glsek), np.mean(temp_evaluate_glsek), np.std(temp_evaluate_glsek)])
        evaluate_uniform1.append([np.max(temp_evaluate_uniform1), np.mean(temp_evaluate_uniform1), np.std(temp_evaluate_uniform1)])

    return eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time

#############################################