from utils import *
from sklearn.tree import DecisionTreeRegressor
from optimizer import *
from predictor_baseline import *
import pandas as pd


def DECART(dataset, metrics, month):

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

    def cart_builder(a, b, c):
        model = DecisionTreeRegressor(
            max_depth=a,
            min_samples_leaf=b,
            min_samples_split=c
        )
        model.fit(train_input, train_output)
        test_predict = model.predict(test_input)
        test_actual = test_output.values
        if metrics == 0:
            return mre_calc(test_predict, test_actual)
        if metrics == 1:
            return sa_calc(test_predict, test_actual, train_output)

    def cart_builder_future(a, b, c):
        model = DecisionTreeRegressor(
            max_depth=a,
            min_samples_leaf=b,
            min_samples_split=c
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(cart_builder_future, metrics, bounds=[(10,20), (1,10), (2,12)])[1]
    # print(config_optimized[0], config_optimized[1], config_optimized[2])
    model_touse = DecisionTreeRegressor(
        max_depth=config_optimized[0],
        min_samples_leaf=config_optimized[1],
        min_samples_split=config_optimized[2]
    )

    model_touse.fit(train_input, train_output)
    test_predict = np.rint(model_touse.predict(test_input))
    test_actual = test_output.values

    result_list_mre = []
    result_list_sa = []

    result_list_mre.append(mre_calc(test_predict, test_actual))
    result_list_sa.append(sa_calc(test_predict, test_actual, train_output))
    # print("pre", test_predict, "act", test_actual)
    if metrics == 0:
        return result_list_mre
    if metrics == 1:
        return result_list_sa

def DECART_bell_test(dataset, metrics, train_df):

    dataset = normalize(dataset)

    validate_train_input = dataset.iloc[:, :-1]
    validate_train_output = dataset.iloc[:, -1]
    validate_test_input = train_df.iloc[:, :-1]
    validate_test_output = train_df.iloc[:, -1]

    def cart_builder_future(a, b, c):
        model = DecisionTreeRegressor(
            max_depth=a,
            min_samples_leaf=b,
            min_samples_split=c
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(cart_builder_future, metrics, bounds=[(10,20), (1,10), (2,12)])[1]
    model_touse = DecisionTreeRegressor(
        max_depth=config_optimized[0],
        min_samples_leaf=config_optimized[1],
        min_samples_split=config_optimized[2]
    )

    model_touse.fit(validate_train_input, validate_train_output)

    return model_touse

def DECART_test(dataset, metrics, month):

    dataset = normalize(dataset)

    for validate_trainset, validate_testset in df_split(dataset, month):
        validate_train_input = validate_trainset.iloc[:, :-1]
        validate_train_output = validate_trainset.iloc[:, -1]
        validate_test_input = validate_testset.iloc[:, :-1]
        validate_test_output = validate_testset.iloc[:, -1]


    def cart_builder_future(a, b, c):
        model = DecisionTreeRegressor(
            max_depth=a,
            min_samples_leaf=b,
            min_samples_split=c
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(cart_builder_future, metrics, bounds=[(10,20), (1,10), (2,12)])[1]
    model_touse = DecisionTreeRegressor(
        max_depth=config_optimized[0],
        min_samples_leaf=config_optimized[1],
        min_samples_split=config_optimized[2]
    )

    model_touse.fit(validate_train_input, validate_train_output)

    return model_touse


def DECART_bellwether(dataset, metrics, month, all_repo, Repo, Directory, goal):

    dataset = normalize(dataset)
    print(dataset.iloc[0:2])

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


    print(validate_test_input.shape,validate_test_output.shape)

    def cart_builder_future(a, b, c):
        model = DecisionTreeRegressor(
            max_depth=a,
            min_samples_leaf=b,
            min_samples_split=c
        )
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(cart_builder_future, metrics, bounds=[(10,20), (1,10), (2,12)])[1]
    model_touse = DecisionTreeRegressor(
        max_depth=config_optimized[0],
        min_samples_leaf=config_optimized[1],
        min_samples_split=config_optimized[2]
    )

    model_touse.fit(validate_train_input, validate_train_output)

    result_list_mre = {}
    result_list_sa = {}

    for test_repo in all_repo:
        test_data = data_goal_arrange(test_repo, Directory, goal)
        test_data = normalize(test_data)

        _, _validate_testset = df_split_month(test_data, month)

        test_input = _validate_testset.iloc[:, :-1]
        test_output = _validate_testset.iloc[:, -1]

        # test_predict = np.rint(model_touse.predict(test_input))
        test_predict = model_touse.predict(test_input)
        test_actual = test_output.values

        if test_repo not in result_list_mre.keys():
            result_list_mre[test_repo] = []
            result_list_sa[test_repo] = []

        mre = mre_calc(test_predict, test_actual)
        # sa = sa_calc(test_predict, test_actual, validate_train_output)
        
        result_list_mre[test_repo] = mre

    if metrics == 0:
        return result_list_mre, model_touse
    if metrics == 1:
        return result_list_sa, model_touse


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

