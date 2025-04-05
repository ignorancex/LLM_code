from experiment.predictor_advance import *
from experiment.utils import *
import os, time


def decart_test(Repo, Directory, metrics, repeats, goal, month):
    data = data_goal_arrange(Repo, Directory, goal)
    list_temp = []
    for _ in range(repeats):
        list_temp.append(DECART(data, metrics, month))
    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())
    print("median for DECART:", np.median(list_output))


def cart_test(Repo, Directory, metrics, repeats, goal, month):
    data = data_goal_arrange(Repo, Directory, goal)
    list_temp = []
    for _ in range(repeats):
        list_temp.append(CART(data, month)[metrics])
    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())
    print("median for CART:", np.median(list_output))


def cart_tuning_test(Repo, Directory, metrics, repeats, goal, month):
    data = data_goal_arrange(Repo, Directory, goal).iloc[:-1]
    list_temp = []
    for _ in range(repeats):
        list_temp.append(CART(data, month, a=16, b=3, c=7)[metrics])
    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())
    print("median for CART-test:", np.median(list_output))


if __name__ == '__main__':

    path = r'../data/data_cleaned/'
    repo = "abp_monthly.csv"
    metrics = 0
    repeats = 1
    goal = 0
    month = 1
    cart_test(repo, path, metrics, repeats, goal, month)
    decart_test(repo, path, metrics, repeats, goal, month)
    cart_tuning_test(repo, path, metrics, repeats, goal, month)
