import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
# import matplotlib.pyplot as plt
from utils import *

def de(fun_opt, metrics, bounds, mut=0.5, crossp=0.5, popsize=20, itrs=10):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds)[:,0], np.asarray(bounds)[:,1]
    diff = np.fabs(min_b - max_b)
    print(diff)
    # pop_denorm = min_b + pop * diff
    pop_denorm = np.rint(min_b + pop * diff).astype(np.int)
    fitness = np.asarray([fun_opt(*ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(itrs):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            # trial_denorm = min_b + trial * diff
            trial_denorm = np.rint(min_b + trial * diff).astype(np.int)
            f = fun_opt(*trial_denorm)
            if metrics == 0:
                if f < fitness[j]:  ####### MRE
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:  ####### MRE
                        best_idx = j
                        best = trial_denorm
            if metrics == 1:
                if f > fitness[j]:  ####### SA
                    fitness[j] = f
                    pop[j] = trial
                    if f > fitness[best_idx]:  ####### SA
                        best_idx = j
                        best = trial_denorm
        return fitness[best_idx], best


def flash(train_input, train_actual_effort, test_input, test_actual_effort, metrics, pop_size):
    def convert(index):  # 12 12 20
        a = int(index / 240 + 1)
        b = int(index % 240 / 20 + 1)
        c = int(index % 20 + 2)
        return a, b, c

    def convert_lr(index):  # 30 2 2 100
        a = int(index / 400 + 1)
        b = int(index % 400 / 200 + 1)
        c = int(index % 200 / 100 + 1)
        d = int(index % 100 + 1)
        return a, b, c, d

    all_case = set(range(0, 2880))
    modeling_pool = random.sample(all_case, pop_size)

    List_X = []
    List_Y = []

    for i in range(len(modeling_pool)):
        temp = convert(modeling_pool[i])
        List_X.append(temp)
        model = DecisionTreeRegressor(max_depth=temp[0], min_samples_leaf=temp[1], min_samples_split=temp[2])
        model.fit(train_input, train_actual_effort)
        test_predict_effort = model.predict(test_input)
        test_predict_Y = test_predict_effort
        test_actual_Y = test_actual_effort.values

        if metrics == 0:
            List_Y.append(mre_calc(test_predict_Y, test_actual_Y))  ######### for MRE
        if metrics == 1:
            List_Y.append(sa_calc(test_predict_Y, test_actual_Y, train_actual_effort))  ######### for SA

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 201 and life > 0:  # eval_number
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if metrics == 0:
            if candi_pred_value < np.median(List_Y):  ######### for MRE
                List_X.append(candidate[0])
                candi_config = candidate[0]
                candi_model = DecisionTreeRegressor(max_depth=candi_config[0], min_samples_leaf=candi_config[1],
                                                    min_samples_split=candi_config[2])
                candi_model.fit(train_input, train_actual_effort)
                candi_pred_Y = candi_model.predict(test_input)
                candi_actual_Y = test_actual_effort.values

                List_Y.append(mre_calc(candi_pred_Y, candi_actual_Y))  ######### for MRE

            else:
                life -= 1

        if metrics == 1:
            if candi_pred_value > np.median(List_Y):  ######### for SA
                List_X.append(candidate[0])
                candi_config = candidate[0]
                candi_model = DecisionTreeRegressor(max_depth=candi_config[0], min_samples_leaf=candi_config[1],
                                                    min_samples_split=candi_config[2])
                candi_model.fit(train_input, train_actual_effort)
                candi_pred_Y = candi_model.predict(test_input)
                candi_actual_Y = test_actual_effort.values

                List_Y.append(sa_calc(candi_pred_Y, candi_actual_Y, train_actual_effort))  ######### for SA

            else:
                life -= 1

    # temp_tree = candi_model
    # tree.plot_tree(temp_tree, feature_names=list(train_input.columns.values))
    # plt.show()

    if metrics == 0:
        return np.min(List_Y)  ########## min for MRE
    if metrics == 1:
        return np.max(List_Y)  ########## min for SA
