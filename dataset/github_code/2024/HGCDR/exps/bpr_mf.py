import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib import layers
import math
import sys
sys.path.append("..")
import utils
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_my_data(select_domain):
    user_id_column_name = 'reviewerID'
    item_id_column_name = 'asin'
    global_item_id_column_name = 'global_item_id'

    data_dir = basedir = '../data_v2'
    task_name = 'task_1'
    data_dir = os.path.join(basedir, task_name)
    dataframe = pd.read_csv(os.path.join(data_dir, 'total_df.csv'))
    if select_domain != None:
        dataframe = dataframe[dataframe['item_domain'] == select_domain]
    dataframe[global_item_id_column_name] = dataframe[item_id_column_name].astype('str') + '_' + dataframe[
        'item_domain'].astype('str')
    global_item = dataframe[global_item_id_column_name].unique()
    global_item_dict = dict(zip(global_item, range(len(global_item))))
    dataframe[global_item_id_column_name] = dataframe[global_item_id_column_name].map(global_item_dict)

    global_user = dataframe[user_id_column_name].unique()
    global_user_dict = dict(zip(global_user, range(len(global_user))))
    dataframe[user_id_column_name] = dataframe[user_id_column_name].map(global_user_dict)

    user_ratings = {}
    for index, row in dataframe.iterrows():
        # print(index, row)
        user_id = row[user_id_column_name]
        global_item_id = row[global_item_id_column_name]
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        user_ratings[user_id].append(global_item_id)
    users = user_ratings.keys()
    movies = global_item_dict.values()
    print('user num: ', len(users), ", item num: ", len(movies), ", rating num: ", len(dataframe))
    return users, movies, user_ratings, len(dataframe)

def load_data(file_path):
    '''导入数据
    input: file_path(string):评分数据文件路径
    output：
        users(list): 用户列表
        movies(list): 电影列表
        user_ratings(dict): [用户]-[看过的电影列表]字典
    '''
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    users = data.index
    movies = data.columns

    user_ratings = defaultdict(dict)
    no = 0
    for i in range(len(users)):
        #         print(list(np.where(data.loc[user] != 0.0)[0]))
        user_movies = list(np.where(data.loc[users[i]] != 0.0)[0])
        if len(user_movies) > 1:
            user_ratings[no] = user_movies
            no += 1

    movies = [int(movie) for movie in movies]
    return users, movies, user_ratings


def generate_test(user_ratings):
    '''随机抽取评分电影i,生成测试数据集
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
    output:
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
    '''
    user_ratings_test = {}
    for user in user_ratings:
        user_ratings_test[user] = random.sample(user_ratings[user], 1)[0]
    return user_ratings_test


def generate_train_batch(user_ratings, user_ratings_test, n, batch_size=512):
    '''构造训练三元组
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
        n(int): 电影数目
        batch_size(int): 批大小
    output:
        trian_batch: 训练批
    '''
    t = []
    user_key = user_ratings.keys()
    for b in range(batch_size):
        u = random.sample(user_key, 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(0, n - 1)
        while j in user_ratings[u]:
            j = random.randint(0, n - 1)

        t.append([u, i, j])

    train_batch = np.asarray(t)
    return train_batch


def generate_test_batch(user_ratings, user_ratings_test, n):
    '''构造训练三元组
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
        movies(list): 电影ID列表
    output:
        test_batch: 测试批
    '''
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(n):
            if j not in user_ratings[u]:
                t.append([u, i, j])
        # print(t)
        yield np.asarray(t)


def BPR(user_ratings, user_ratings_test, movies, k, beta, learning_rate, training_epochs, display_step=1):
    '''利用梯度下降法求解BPR
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
        movies(list): 电影ID列表
        k(int): 分解矩阵的参数
        beta(float): 正则化参数
        learning_rate(float): 学习率
        training_epochs(int): 最大迭代次数
    output:
        U, V: 分解后的矩阵
    '''

    m = len(user_ratings)
    n = len(movies)

    # 1.初始化变量
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    U = tf.get_variable("U", [m, k], initializer=tf.random_normal_initializer(0, 0.1))
    V = tf.get_variable("V", [n, k], initializer=tf.random_normal_initializer(0, 0.1))

    u_emb = tf.nn.embedding_lookup(U, u)
    i_emb = tf.nn.embedding_lookup(V, i)
    j_emb = tf.nn.embedding_lookup(V, j)

    # 3.构建模型
    pred = tf.reduce_mean(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)
    auc = tf.reduce_mean(tf.to_float(pred > 0))
    regu = layers.l2_regularizer(beta)(u_emb)
    regi = layers.l2_regularizer(beta)(i_emb)
    regj = layers.l2_regularizer(beta)(j_emb)

    # cost = regu + regi + regj - tf.reduce_mean(tf.log(tf.sigmoid(pred)))
    cost = - tf.reduce_mean(tf.log(tf.sigmoid(pred)))

    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # 4.进行训练
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(init)
        batch_size = 512
        total_batch = math.ceil(rating_num / batch_size)
        print("total_batch: ", total_batch)
        for epoch in range(training_epochs):
            start_time = time.time()
            avg_cost = 0
            avg_auc = 0
            # for p in range(1, 100):
            for p in range(1, total_batch):
                uij = generate_train_batch(user_ratings, user_ratings_test, n, batch_size)
                batch_u_emb, batch_i_emb, batch_j_emb, batch_pred, batch_cost, batch_auc, _ = sess.run([u_emb, i_emb, j_emb, pred, cost, auc, train_step], feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
                # print(batch_u_emb, batch_i_emb, batch_j_emb, batch_pred, batch_cost, batch_auc)
                # print(batch_cost, batch_auc)
                avg_cost += batch_cost
                avg_auc += batch_auc
                # print('train auc: auc=%.4f, loss=%.4f' % (batch_auc, batch_cost))

            # 打印cost
            if (epoch + 1) % display_step == 0:
                # 计算准确度
                user_count = 0
                test_avg_auc = 0
                test_avg_cost = 0

                for t_uij in generate_test_batch(user_ratings, user_ratings_test, n):
                    test_batch_auc, test_batch_cost = sess.run([auc, cost], feed_dict={u: t_uij[:, 0], i: t_uij[:, 1],
                                                                                       j: t_uij[:, 2]})
                    user_count += 1
                    test_avg_auc += test_batch_auc
                    test_avg_cost += test_batch_cost
                    # print('test auc: auc=%.4f' % (test_batch_auc))
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost / p), "auc=", "{:.9f}".format(avg_auc / p), "test_cost",
                      "{:.9f}".format(test_avg_cost / user_count), "test_auc", "{:.9f}".format(test_avg_auc / user_count), "time", "{:.9f}".format(time.time() - start_time))

        # 打印变量
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
            print(v)

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, "model/bpr_t/t")
        print("Optimization Finished!")


if __name__ == "__main__":
    utils.tf_seed_setting(7)
    users, movies, user_ratings, rating_num = load_my_data("instrument")
    # 参数
    k = 20
    beta = 0.00000001
    learning_rate = 0.001
    training_epochs = 1000
    display_step = 1

    user_ratings_test = generate_test(user_ratings)

    BPR(user_ratings, user_ratings_test, movies, k, beta, learning_rate, training_epochs, display_step)