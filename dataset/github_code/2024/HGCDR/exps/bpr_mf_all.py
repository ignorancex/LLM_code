import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers
import math
import time
import sys
sys.path.append("..")
import utils
import copy
import heapq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_my_data(select_domain):
    user_id_column_name = 'reviewerID'
    item_id_column_name = 'asin'
    global_item_id_column_name = 'global_item_id'

    basedir = '../data_v2'
    task_name = 'task_1'
    data_dir = os.path.join(basedir, task_name)
    dataframe = pd.read_csv(os.path.join(data_dir, 'total_df.csv'))

    if select_domain in ["instrument", "music", "video"]:
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
    domain_user_ratings = {}
    domain_items = {}
    domain_ratings_num = {}
    for index, row in dataframe.iterrows():
        # print(index, row)
        user_id = row[user_id_column_name]
        global_item_id = row[global_item_id_column_name]
        domain = row['item_domain']
        if domain not in domain_user_ratings:
            domain_user_ratings[domain] = {}
        if domain not in domain_items:
            domain_items[domain] = set()
        if domain not in domain_ratings_num:
            domain_ratings_num[domain] = 0
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        if user_id not in domain_user_ratings[domain]:
            domain_user_ratings[domain][user_id] = []
        user_ratings[user_id].append(global_item_id)
        domain_user_ratings[domain][user_id].append(global_item_id)
        domain_items[domain].add(global_item_id)
        domain_ratings_num[domain] += 1
    users = user_ratings.keys()
    items = global_item_dict.values()
    print('user num: ', len(users), ", item num: ", len(items), ", rating num: ", len(dataframe))
    domain_items['all'] = set(items)
    domain_user_ratings['all'] = user_ratings
    domain_ratings_num['all'] = len(dataframe)
    return users, domain_items, domain_user_ratings, domain_ratings_num


def generate_test(domain_user_ratings):
    '''随机抽取评分电影i,生成测试数据集
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
    output:
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
    '''
    user_ratings_test = {}
    domain_user_ratings_train = copy.deepcopy(domain_user_ratings)
    domain_user_ratings_test = {}
    for domain in domain_user_ratings:
        if domain == 'all':
            continue
        domain_user_ratings_test[domain] = {}
        user_ratings = domain_user_ratings[domain]
        for user in user_ratings:
            item_id = random.sample(user_ratings[user], 1)[0]
            domain_user_ratings_test[domain][user] = item_id
            # user_ratings_test[user] = item_id
            domain_user_ratings_train[domain][user].remove(item_id)
            domain_user_ratings_train['all'][user].remove(item_id)
    # domain_user_ratings_test['all'] = user_ratings_test
    return domain_user_ratings_train, domain_user_ratings_test

def generate_neg_test(domain_user_ratings, domain_items, neg_num = 99):
    domain_user_ratings_neg_test = {}
    for domain in domain_user_ratings:
        if domain == 'all':
            continue
        domain_user_ratings_neg_test[domain] = {}
        user_ratings = domain_user_ratings[domain]
        item_set = domain_items[domain]
        for user in user_ratings:
            neg_num_list = set()
            while len(neg_num_list) < neg_num:
                neg_item = random.sample(item_set, 1)[0]
                if neg_item not in user_ratings[user] and neg_item not in neg_num_list:
                    neg_num_list.add(neg_item)
            domain_user_ratings_neg_test[domain][user] = neg_num_list
    return domain_user_ratings_neg_test


def generate_train_batch(user_ratings, user_ratings_train, user_ratings_test, items, batch_size=512):
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
    user_key = user_ratings_train.keys()
    # batch_u = random.sample(user_key, batch_size)
    for b in range(batch_size):
        u = random.sample(user_key, 1)[0]
        i = random.sample(user_ratings_train[u], 1)[0]
        # while i == user_ratings_test[u]:
        #     i = random.sample(user_ratings[u], 1)[0]

        j = random.sample(items, 1)[0]
        rating_set = set(user_ratings[u])
        while j in rating_set:
            j = random.sample(items, 1)[0]

        t.append([u, i, j])

    train_batch = np.asarray(t)
    return train_batch


def generate_test_batch(user_ratings, user_ratings_test, items_set):
    '''构造训练三元组
    input:
        user_ratings(dict): [用户]-[看过的电影列表]字典
        user_ratings_test(dict): [用户]-[一部看过的电影i]字典
        movies(list): 电影ID列表
    output:
        test_batch: 测试批
    '''
    for u in user_ratings:
        t = []
        i = user_ratings_test[u]
        for j in items_set:
            if j not in user_ratings[u]:
                t.append([u, i, j])
        # print(t)
        yield np.asarray(t)

def generate_test_batch_new(user_ratings_test, user_ratings_neg_test):
    for u in user_ratings_test:
        t = []
        i = user_ratings_test[u]
        t.append([u, i])
        for j in user_ratings_neg_test[u]:
            t.append([u, j])
        yield np.asarray(t)

def eval_auc():
    pass

def eval_hit():
    pass


def BPR(domain_user_ratings, domain_user_ratings_train, domain_user_ratings_test, domain_items, k, beta, learning_rate, training_epochs, display_step=1, domain = 'all'):
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

    m = len(domain_user_ratings['all'])
    n = len(domain_items['all'])

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
    score_ui = tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keepdims=True)
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
        total_batch = math.ceil(domain_ratings_num[domain] / batch_size)
        print("total_batch: ", total_batch)

        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        for epoch in range(training_epochs):
            t0 = time.time()
            avg_cost = 0
            avg_auc = 0

            for p in range(1, total_batch):
                uij = generate_train_batch(domain_user_ratings[domain], domain_user_ratings_train[domain], None, domain_items[domain], batch_size)
                batch_u_emb, batch_i_emb, batch_j_emb, batch_pred, batch_cost, batch_auc, _ = sess.run([u_emb, i_emb, j_emb, pred, cost, auc, train_step], feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
                # print(batch_u_emb, batch_i_emb, batch_j_emb, batch_pred, batch_cost, batch_auc)
                # print(batch_cost, batch_auc)
                avg_cost += batch_cost
                avg_auc += batch_auc
                # print('train auc: auc=%.4f, loss=%.4f' % (batch_auc, batch_cost))

            # 打印cost
            if (epoch + 1) % display_step == 0:
                if domain == 'all':
                    for key in domain_user_ratings_test.keys():
                        user_ratings = domain_user_ratings[key]
                        user_ratings_test = domain_user_ratings_test[key]
                        user_ratings_neg_test = domain_user_ratings_neg_test[key]
                        """
                        user_count = 0
                        test_avg_auc = 0
                        test_avg_cost = 0
                        if key == 'all':
                            continue
                        items_set = set(domain_items[key])
                        for t_uij in generate_test_batch(user_ratings, user_ratings_test, items_set):
                            test_batch_auc, test_batch_cost = sess.run([auc, cost],
                                                                       feed_dict={u: t_uij[:, 0], i: t_uij[:, 1],
                                                                                  j: t_uij[:, 2]})
                            user_count += 1
                            test_avg_auc += test_batch_auc
                            test_avg_cost += test_batch_cost
                            # print('test auc: auc=%.4f' % (test_batch_auc))
                        print("Epoch:", '%04d' % (epoch + 1), key, "cost=", "{:.9f}".format(avg_cost / p), "auc=",
                              "{:.9f}".format(avg_auc / p), "test_cost",
                              "{:.9f}".format(test_avg_cost / user_count), "test_auc",
                              "{:.9f}".format(test_avg_auc / user_count), "time=", "{:.9f}".format(time.time() - t0))
                        """
                        user_count = 0
                        test_avg_hr = []
                        test_avg_ngcg = []
                        for t_uij in generate_test_batch_new(user_ratings_test, user_ratings_neg_test):
                            batch_score_ui = sess.run(score_ui, feed_dict={u: t_uij[:, 0], i: t_uij[:, 1]})
                            item_score_dict = {}
                            for k in range(len(t_uij[:, 1])):
                                item_score_dict[t_uij[k, 1]] = batch_score_ui[k]
                            ranklist = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)
                            tmp_hr = getHitRatio(ranklist, t_uij[0, 1])
                            tmp_NDCG = getNDCG(ranklist, t_uij[0, 1])
                            test_avg_hr.append(tmp_hr)
                            test_avg_ngcg.append(tmp_NDCG)
                        print("Epoch:", '%04d' % (epoch + 1), key, "cost=", "{:.9f}".format(avg_cost / p), "auc=",
                              "{:.9f}".format(avg_auc / p), "HR=", "{:.9f}".format(np.mean(test_avg_hr)), "NDCG=",
                              "{:.9f}".format(np.mean(test_avg_ngcg)), "time=", "{:.9f}".format(time.time() - t0))
                else:
                    user_ratings = domain_user_ratings[domain]
                    user_ratings_test = domain_user_ratings_test[domain]
                    user_ratings_neg_test = domain_user_ratings_neg_test[domain]
                    # 评估auc
                    """
                    user_count = 0
                    test_avg_auc = 0
                    test_avg_cost = 0
                    items_set = set(domain_items[domain])
                    for t_uij in generate_test_batch(user_ratings, user_ratings_test, items_set):
                        test_batch_auc, test_batch_cost = sess.run([auc, cost],
                                                                   feed_dict={u: t_uij[:, 0], i: t_uij[:, 1],
                                                                              j: t_uij[:, 2]})
                        user_count += 1
                        test_avg_auc += test_batch_auc
                        test_avg_cost += test_batch_cost

                    print("Epoch:", '%04d' % (epoch + 1), domain, "cost=", "{:.9f}".format(avg_cost / p), "auc=",
                          "{:.9f}".format(avg_auc / p), "test_cost",
                          "{:.9f}".format(test_avg_cost / user_count), "test_auc",
                          "{:.9f}".format(test_avg_auc / user_count), "time=", "{:.9f}".format(time.time() - t0))
                    """


                    user_count = 0
                    test_avg_hr = []
                    test_avg_ngcg = []
                    for t_uij in generate_test_batch_new(user_ratings_test, user_ratings_neg_test):
                        batch_score_ui = sess.run(score_ui, feed_dict={u: t_uij[:, 0], i: t_uij[:, 1]})
                        item_score_dict = {}
                        for k in range(len(t_uij[:, 1])):
                            item_score_dict[t_uij[k, 1]] = batch_score_ui[k]
                        ranklist = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)
                        tmp_hr = getHitRatio(ranklist, t_uij[0,1])
                        tmp_NDCG = getNDCG(ranklist, t_uij[0,1])
                        test_avg_hr.append(tmp_hr)
                        test_avg_ngcg.append(tmp_NDCG)
                    print("Epoch:", '%04d' % (epoch + 1), domain, "cost=", "{:.9f}".format(avg_cost / p), "auc=",
                          "{:.9f}".format(avg_auc / p), "HR=", "{:.9f}".format(np.mean(test_avg_hr)), "NDCG=",
                            "{:.9f}".format(np.mean(test_avg_ngcg)), "time=", "{:.9f}".format(time.time() - t0))
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
    domain = sys.argv[1]
    # domain = 'instrument'
    users, domain_items, domain_user_ratings, domain_ratings_num = load_my_data(domain)
    # 参数
    k = 20
    beta = 0.00000001
    learning_rate = 0.001
    training_epochs = 1000
    display_step = 1

    domain_user_ratings_train, domain_user_ratings_test = generate_test(domain_user_ratings)
    domain_user_ratings_neg_test = generate_neg_test(domain_user_ratings, domain_items)

    BPR(domain_user_ratings, domain_user_ratings_train, domain_user_ratings_test, domain_items, k, beta, learning_rate, training_epochs, display_step, domain)