from utils import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from optimizer import *
from predictor_baseline import *
import pandas as pd

from learners import SK_DTR, SK_RRF
from DE import DE_Tune_ML
import CFS
import utils



def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
    return tuner.Tune()


def DECART(dataset, metrics, month):

    dataset = normalize(dataset)

    trainset, testset = df_split_month(dataset, month)

    train_input = trainset.iloc[:, :-1]
    train_output = trainset.iloc[:, -1]
    test_input = testset.iloc[:, :-1]
    test_output = testset.iloc[:, -1]

    validation_trainset, valdation_testset = df_split_month(trainset, month)

    validate_train_input = validation_trainset.iloc[:, :-1]
    validate_train_output = validation_trainset.iloc[:, -1]
    validate_test_input = valdation_testset.iloc[:, :-1]
    validate_test_output = valdation_testset.iloc[:, -1]

    model = [SK_DTR][0]
    params, evaluation = tune_learner(model, validate_train_input, validate_train_output, validate_test_input,
                                            validate_test_output, metrics)
    model_touse = DecisionTreeRegressor()
    model_touse.set_params(**params)

    model_touse.fit(validate_train_input, validate_train_output)

    result_list_mre = {}
    result_list_sa = {}

    test_predict = model_touse.predict(test_input)
    test_actual = test_output.values

    mre = mre_calc(test_predict, test_actual)

    return mre, model_touse





def DECART_test(dataset, metrics, month):

    dataset = normalize(dataset)

    validate_trainset, validate_testset = df_split_month(dataset, month)
    validate_train_input = validate_trainset.iloc[:, :-1]
    validate_train_output = validate_trainset.iloc[:, -1]
    validate_test_input = validate_testset.iloc[:, :-1]
    validate_test_output = validate_testset.iloc[:, -1]

    model = [SK_DTR][0]
    params, evaluation = tune_learner(model, validate_train_input, validate_train_output, validate_test_input,
                                            validate_test_output, metrics)
    model_touse = DecisionTreeRegressor()
    model_touse.set_params(**params)

    model_touse.fit(validate_train_input, validate_train_output)

    return model_touse


def DECART_bellwether(dataset, metrics, month, all_repo, Repo, Directory, goal):

    dataset = normalize(dataset)

    validate_trainset, validate_testset = df_split_month(dataset, month)

    validate_train_input = validate_trainset.iloc[:, :-1]
    validate_train_output = validate_trainset.iloc[:, -1]
    validate_test_input = validate_testset.iloc[:, :-1]
    validate_test_output = validate_testset.iloc[:, -1]

    for test_repo in all_repo:
        test_data = data_goal_arrange(test_repo, Directory, goal)
        test_data = normalize(test_data) 

        _validate_testset, _ = df_split_month(test_data, month)

        test_input = _validate_testset.iloc[:, :-1]
        test_output = _validate_testset.iloc[:, -1]

        validate_test_input = pd.concat([validate_test_input,test_input], axis = 0)
        validate_test_output = pd.concat([validate_test_output,test_output], axis = 0)


    # model = [SK_DTR][0]
    # params, evaluation = tune_learner(model, validate_train_input, validate_train_output, validate_test_input,
    #                                         validate_test_output, metrics)
    model_touse = RandomForestRegressor()
    model_touse.set_params()

    model_touse.fit(validate_train_input, validate_train_output)

    result_list_mre = {}
    result_list_sa = {}

    for test_repo in all_repo:
        test_data = data_goal_arrange(test_repo, Directory, goal)

        test_data = normalize(test_data)

        _, _validate_testset = df_split_month(test_data, month)

        test_input = _validate_testset.iloc[:, :-1]
        test_output = _validate_testset.iloc[:, -1]

        test_predict = model_touse.predict(test_input)
        test_actual = test_output.values

        if test_repo not in result_list_mre.keys():
            result_list_mre[test_repo] = []
            result_list_sa[test_repo] = []

        mre = mre_calc(test_predict, test_actual)
        sa = sa_calc(test_predict, test_actual, validate_train_output)
        
        result_list_mre[test_repo] = mre

    if metrics == 0:
        return result_list_mre, model_touse, sa
    if metrics == 1:
        return result_list_sa, model_touse, sa

def DECART_bellwether_CFS(dataset, metrics, month, all_repo, Repo, Directory, goal, cols):

    dataset = normalize(dataset)

    validate_trainset, validate_testset = df_split_month(dataset, month)

    validate_train_input = validate_trainset.iloc[:, :-1]
    validate_train_output = validate_trainset.iloc[:, -1]
    validate_test_input = validate_testset.iloc[:, :-1]
    validate_test_output = validate_testset.iloc[:, -1]

    # for test_repo in all_repo:
    #     test_data = data_goal_arrange(test_repo, Directory, goal)
    #     test_data = test_data[cols]
    #     test_data = normalize(test_data) 
    #     _validate_testset, _ = df_split_month(test_data, month)

    #     test_input = _validate_testset.iloc[:, :-1]
    #     test_output = _validate_testset.iloc[:, -1]

    #     validate_test_input = pd.concat([validate_test_input,test_input], axis = 0)
    #     validate_test_output = pd.concat([validate_test_output,test_output], axis = 0)


    model = [SK_DTR][0]
    params, evaluation = tune_learner(model, validate_train_input, validate_train_output, validate_test_input,
                                            validate_test_output, metrics)
    model_touse = DecisionTreeRegressor()
    model_touse.set_params(**params)

    model_touse.fit(validate_train_input, validate_train_output)

    result_list_mre = {}
    result_list_sa = {}

    for test_repo in all_repo:
        test_data = data_goal_arrange(test_repo, Directory, goal)
        test_data = test_data[cols]
        test_data = normalize(test_data)

        _, _validate_testset = df_split_month(test_data, month)

        test_input = _validate_testset.iloc[:, :-1]
        test_output = _validate_testset.iloc[:, -1]

        test_predict = model_touse.predict(test_input)
        test_actual = test_output.values

        if test_repo not in result_list_mre.keys():
            result_list_mre[test_repo] = []
            result_list_sa[test_repo] = []

        mre = mre_calc(test_predict, test_actual)
        sa = sa_calc(test_predict, test_actual, validate_train_output)
        
        result_list_mre[test_repo] = mre

    if metrics == 0:
        return result_list_mre, model_touse, sa
    if metrics == 1:
        return result_list_sa, model_touse, sa


def DERFT(dataset, metrics, month):

    dataset = normalize(dataset)

    for trainset, testset in df_split(dataset, month):
        train_input = trainset.iloc[:, :-1]
        train_output = trainset.iloc[:, -1]
        test_input = testset.iloc[:, :-1]
        test_output = testset.iloc[:, -1]

    for validate_trainset, validate_testset in df_split(trainset, 1):
        validate_train_input = validate_trainset.iloc[:, :-1]
        validate_train_output = validate_trainset.iloc[:, -1]
        validate_test_input = validate_testset.iloc[:, :-1]
        validate_test_output = validate_testset.iloc[:, -1]

    def rft_builder(a, b, c, d, e):
        model = RandomForestRegressor(
            n_estimators=a,
            max_depth=b,
            min_samples_split=c,
            min_samples_leaf=d,
            max_leaf_nodes=e
        )
        model.fit(train_input, train_output)
        test_predict = model.predict(test_input)
        test_actual = test_output.values
        if metrics == 0:
            return mre_calc(test_predict, test_actual)
        if metrics == 1:
            return sa_calc(test_predict, test_actual, train_output)

    def rft_builder_future(a, b, c, d, e):
        model = RandomForestRegressor(
            n_estimators=a,
            max_depth=b,
            min_samples_split=c,
            min_samples_leaf=d,
            max_leaf_nodes=e
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(rft_builder, metrics, bounds=[(10, 100), (1, 10), (2, 20), (1, 10), (10, 20)])[1]
    # print(config_optimized[0], config_optimized[1], config_optimized[2])
    model_touse = RandomForestRegressor(
        n_estimators = config_optimized[0],
        max_depth = config_optimized[1],
        min_samples_split = config_optimized[2],
        min_samples_leaf = config_optimized[3],
        max_leaf_nodes = config_optimized[4]
    )

    model_touse.fit(train_input, train_output)
    test_predict = np.rint(model_touse.predict(test_input))
    test_actual = test_output.values

    result_list_mre = []
    result_list_sa = []
    # print("DECART", "predict", test_predict, "actual", test_actual)
    result_list_mre.append(mre_calc(test_predict, test_actual))
    result_list_sa.append(sa_calc(test_predict, test_actual, train_output))

    if metrics == 0:
        return result_list_mre
    if metrics == 1:
        return result_list_sa


def DEKNN(dataset, metrics, month):

    dataset = normalize(dataset)

    for trainset, testset in df_split(dataset, month):
        train_input = trainset.iloc[:, :-1]
        train_output = trainset.iloc[:, -1]
        test_input = testset.iloc[:, :-1]
        test_output = testset.iloc[:, -1]

    for validate_trainset, validate_testset in df_split(trainset, 1):
        validate_train_input = validate_trainset.iloc[:, :-1]
        validate_train_output = validate_trainset.iloc[:, -1]
        validate_test_input = validate_testset.iloc[:, :-1]
        validate_test_output = validate_testset.iloc[:, -1]

    def knn_builder(a, b, c):
        model = neighbors.KNeighborsRegressor(
            n_neighbors=a,
            leaf_size=b,
            p=c
        )
        model.fit(train_input, train_output)
        test_predict = model.predict(test_input)
        test_actual = test_output.values
        if metrics == 0:
            return mre_calc(test_predict, test_actual)
        if metrics == 1:
            return sa_calc(test_predict, test_actual, train_output)

    def knn_builder_future(a, b, c):
        model = neighbors.KNeighborsRegressor(
            n_neighbors=a,
            leaf_size=b,
            p=c
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(knn_builder, metrics, bounds=[(1, 5), (10, 50), (1, 3)])[1]
    # print(config_optimized[0], config_optimized[1], config_optimized[2])
    model_touse = neighbors.KNeighborsRegressor(
        n_neighbors=config_optimized[0],
        leaf_size=config_optimized[1],
        p=config_optimized[2]
    )

    model_touse.fit(train_input, train_output)
    test_predict = np.rint(model_touse.predict(test_input))
    test_actual = test_output.values

    result_list_mre = []
    result_list_sa = []
    # print("DECART", "predict", test_predict, "actual", test_actual)
    result_list_mre.append(mre_calc(test_predict, test_actual))
    result_list_sa.append(sa_calc(test_predict, test_actual, train_output))

    if metrics == 0:
        return result_list_mre
    if metrics == 1:
        return result_list_sa


if __name__ == '__main__':

    path = r'../data/data_cleaned/'
    repo = "joker_monthly.csv"

    metrics = 0  # "0" for MRE, "1" for SA
    repeats = 1
    goal = 1
    month = 1

    data = data_goal_arrange(repo, path, goal)

    list_temp = []
    for way in range(repeats):
        list_temp.append(DECART(data, metrics, month))

    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())

    print(list_output)
    print("median DECART:", np.median(list_output))

    list_temp = []
    for way in range(repeats):
        list_temp.append(CART(data, month)[metrics])

    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())

    print(list_output)
    print("median CART:", np.median(list_output))

