import numpy as np
import scipy.special

# sum of squares in a list
def square_sum(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]*list[i]
    return sum

#################################################
# exchange the formulation of coreset
def coreset_nt_and_divide(T,coreset,type):
    if type == 0:
        temp = []
        for i in range(len(coreset)):
            time = coreset[i][0] % T
            id = int((coreset[i][0]-time)/T)
            temp.append([id,time,coreset[i][1]])
    
        coreset_unify = [[temp[0][0],[temp[0][1],temp[0][2]]]]
        id = temp[0][0]
        count = 0
        for i in range(len(coreset)):
            if temp[i][0] == id:
                coreset_unify[count].append([temp[i][1], temp[i][2]])
            else:
                id = temp[i][0]
                count += 1
                coreset_unify.append([temp[i][0],[temp[i][1],temp[i][2]]])
        return coreset_unify
    
    if type == 1:
        coreset_unify = []
        for i in range(len(coreset)):
            for t in range(1,len(coreset[i])):
                position = coreset[i][0] * T + coreset[i][t][0]
                coreset_unify.append([position,coreset[i][t][1]])
        return coreset_unify



###############################################
# uniform sampling for glse
def uniform_1(size,nt_sample):
    id_list = [i for i in range(size)]
    weight = [0 for i in range(size)]
    sample = np.random.choice(id_list, size=nt_sample)
    for i in range(nt_sample):
        weight[sample[i]] += size/nt_sample

    # the coreset form
    coreset = []
    for i in range(size):
        if weight[i] > 0:
            coreset.append([i,weight[i]])
    return coreset

################################################
# uniform sampling for glsek
def uniform_2(N,T,n_sample,t_sample):
    # sample individuals
    I_S = uniform_1(N,n_sample)

    # the coreset form
    coreset = []
    for i in range(len(I_S)):
        id = I_S[i]
        coreset.append([id[0]])
        coreset_time = uniform_1(T,t_sample)
        for t in range(len(coreset_time)):
            coreset[i].append([coreset_time[t][0], id[1] * coreset_time[t][1]])

    return coreset

################################################
# sensitivity function for glse
def sen_glse(panel,q,lam):
    # construct matrix Z
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    Z = panel.reshape(N * T, d)
    # construct the column basis of Z
    col_Z = scipy.linalg.orth(Z)

    # sensitivity function for OLSE
    seno = [0 for i in range(len(Z))]
    for i in range(len(Z)):
        seno[i] = square_sum(col_Z[i])

    # sensitivity function for glse
    sen = [0 for i in range(len(Z))]
    for i in range(len(Z)):
        temp = seno[i]
        time = i % T
        index = min(time, q)
        for j in range(index):
            temp += seno[i - j - 1]
        # temp = 2 * temp / lam
        sen[i] = min(1, temp)
    
    return sen
    
################################################
# coreset construction for glse
def coreset_glse(sen,nt_sample):
    size = len(sen)
    # total sensitivity
    total_sen = sum(sen)
    # sampling distribution
    pr = [0 for i in range(size)]
    for i in range(size):
        pr[i] = sen[i]/total_sen

    # importance sampling
    id_list = [i for i in range(size)]
    weight = [0 for i in range(size)]
    sample = np.random.choice(id_list, size=nt_sample, p=pr)
    for i in range(nt_sample):
        weight[sample[i]] += 1/(pr[sample[i]]*nt_sample)

    # the coreset form
    coreset = []
    for i in range(size):
        if weight[i] > 0:
            coreset.append([i,weight[i]])
            #T_id = i % T
            #N_id = int((i-T_id)/T)
            #coreset.append([N_id,T_id,weight[i]])
    return coreset

############################################################
# coreset construction for glsek
def coreset_glsek(panel,q,lam,n_sample,t_sample):
    # construct SVD decomposition of matrix Z_i
    N = panel.shape[0]
    Z = [0 for i in range(N)]
    col_Z = [0 for i in range(N)]
    lam_Z = [0 for i in range(N)]
    U = [0 for i in range(N)]
    L = [0 for i in range(N)]
    for i in range(N):
        Z[i] = np.array([panel[i]])
        col_Z[i], lam_Z[i], row = np.linalg.svd(panel[i])
        lam_Z[i] = np.square(lam_Z[i])
        # lam_Z[i] = np.abs(lam_Z[i])
        U[i] = np.max(lam_Z[i])
        L[i] = np.min(lam_Z[i])

    # sensitivity function for OLSE_k
    seno = [0 for i in range(N)]
    low_sen = np.sum(L)
    for i in range(N):
        low_temp = low_sen + U[i] - L[i]
        seno[i] = U[i]/low_temp

    # sensitivity function for glsek
    sen = [min(1,seno[i]) for i in range(N)]
    print(np.max(sen), np.min(sen))

    # sample individuals
    I_S = coreset_glse(sen,n_sample)
    sen_time = []
    # totalsen_time = []
    # for i in range(len(I_S)):
    #     sen_time.append(sen_glse(Z[I_S[i][0]],q,lam))
    #     totalsen_time.append(sum(sen_time[i]))
    # avg = sum(totalsen_time)/len(I_S)
    # for i in range(len(totalsen_time)):
    #     totalsen_time[i] /= avg

    # sensitivity function for each selected individual
    coreset = []
    for i in range(len(I_S)):
        id = I_S[i]
        coreset.append([id[0]])
        coreset_time = coreset_glse(sen_glse(Z[id[0]],q,lam), t_sample)
        for t in range(len(coreset_time)):
            coreset[i].append([coreset_time[t][0],float(id[1]*coreset_time[t][1])])

    return coreset

##############################################

