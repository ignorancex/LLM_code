from main import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def exp1_plot_images_of_output_centers(k, X_test, output, l_centers, random_centers, same_center):
    """
    exp 1: MNIST. Visualize the difference between the output of SKM, the offline algorithm and random centers.
    NOT USED IN THE PAPER.

    :param k: int. number of centers
    :param X_test:
    :param output:
    :param l_centers:
    :param random_centers:
    :param same_center:
    :return:
    """
    # random_centers = X_train[init_centers(X_train, k)] # random centers
    # same_center = np.zeros((k, l_centers.shape[1]))  # k copies of the origin point
    print('This is the inertia of the output on the test data: %s' % calculate_inertia(X_test, output))
    print('This is the inertia of the offline algorithm on the test data : %s' % calculate_inertia(X_test, l_centers))
    print('This is the inertia of random on the test data: %s' % calculate_inertia(X_test, random_centers))
    print('This is the inertia of same center on the test data: %s' % calculate_inertia(X_test, same_center))
    for index, image in enumerate(output):
        plt.subplot(3, k, index + 1)
        plt.axis('off')
        plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('output')

    for index, image in enumerate(l_centers):
        plt.subplot(3, k, 16 + (index + 1))
        plt.axis('off')
        plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('offline')

    for index, image in enumerate(random_centers):
        plt.subplot(3, k, 32 + (index + 1))
        plt.axis('off')
        plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('random')

    plt.show()


def mnist_prepare():
    """
    Pre-proccess of mnist. Read the data set and remove label.
    :return: x_train, x_test
    """
    mnist_train = pd.read_csv("mnist_train.csv")  # .iloc[:5000]
    mnist_test = pd.read_csv("mnist_test.csv")
    x_train = mnist_train.drop('label', axis=1).values
    x_test = mnist_test.drop('label', axis=1).values
    return x_train, x_test


def forest_data_prepare():
    """
    Pre-process of forest_data. Read the data set and remove label.
    :return: x_train, x_test
    """
    covtype_train = pd.read_csv("covtype.csv")  # .iloc[:50000]
    covtype_train = covtype_train.drop('Cover_Type', axis=1).values
    x_train, x_test = train_test_split(covtype_train, train_size=0.1, test_size=0.01, random_state=42)
    return x_train, x_test


def census1990_prepare():
    """
    Pre-process of census1990. Read the data set and prepare the data (missing feature and etc.)
    :return: x_train, x_test
    """
    census = pd.read_csv("housing.csv")
    census = pd.get_dummies(census)
    pd.set_option('display.max_columns', 500)
    print(census.describe())
    census = census.drop('median_house_value', axis=1)
    median = census['total_bedrooms'].median()
    census['total_bedrooms'].fillna(median, inplace=True)
    x_train, x_test = train_test_split(census, test_size=0.1, random_state=42)
    return x_train, x_test


def to_csv(data):
    """
    Export the data into csv file with name that includes the data
    :param data: df
    :return:
    """
    date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')
    log_file = './csv_output/' + 'fig_' + date + '.csv'
    # np_to_csv = np.vstack((np_number_of_samples, np_rate))
    df_to_csv = pd.DataFrame(data)
    df_to_csv.to_csv(log_file, index=False)


def scale_data(x_train, x_test):
    """
    Scale the data and then apply PCA on the data
    :param x_train:
    :param x_test:
    :return: dataframe
    """
    # scaler = StandardScaler()  # scale to zero mean 1 variance
    scaler = MinMaxScaler()  # scale each feature to (0,1)
    # Fit on training set only.
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # Make an instance of the PCA Model
    pca = PCA(.95)
    # fit PCA on the training set set only
    pca.fit(x_train)
    return pca.transform(x_train), pca.transform(x_test)


def list_q(l_number_of_samples, delta):
    """
    return the q for each sample size in l_number_of_samples
    :param l_number_of_samples:
    :param k: int. number of centers
    :param delta: int.
    :return: list of floats.
    """
    list_of_q = []
    for m in l_number_of_samples:
        list_of_q.append(calc_q(int(m), delta))  # calculate q for each m size
    return np.array(list_of_q)


def exp2(x_train, x_test, k, delta, number_of_iteration_per_m, l_sizeof_m_to_check, birch_kmedoids_mode):
    """
    calculate the mean risk of SKM and the offline algorithms over number_of_iteration_per_m iteration.
    :param x_train: df. train set
    :param x_test: df. test set
    :param k: int - number of centers
    :param delta: int
    :param number_of_iteration_per_m:  int
    :param l_sizeof_m_to_check: list of int.
    :return: numpy_array,numpy_array. np_rate between skm and offline, offline_mean_risk_per_m
    """
    random_state = 0
    offline_mean_risk_per_m = []
    output_mean_risk_per_m = []
    for sample_size in l_sizeof_m_to_check:  # for each size of m
        l_risk_of_output = []
        l_risk_of_offline = []
        for _ in range(number_of_iteration_per_m):  # number of experiments for each m.
            random_state += 1
            np.random.RandomState(seed=random_state)
            x_train_temp = x_train[
                np.random.choice(x_train.shape[0], int(sample_size), replace=False)]  # Shuffles the data.
            output, l_centers = skm1(x_train_temp, k, delta, birch_kmedoids_mode)
            l_risk_of_output.append(calculate_inertia(x_test, output))  # risk of skm
            l_risk_of_offline.append(calculate_inertia(x_test, l_centers))  # risk of offline algorithm
        offline_mean_risk_per_m.append(sum(l_risk_of_offline) / len(l_risk_of_offline))  # mean of the offline risk
        output_mean_risk_per_m.append(sum(l_risk_of_output) / len(l_risk_of_output))  # mean of the skm risk

    np_rate = np.divide(output_mean_risk_per_m, offline_mean_risk_per_m).transpose()  # the rate between both risks.
    return np_rate, offline_mean_risk_per_m


def exp_mnist(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker):
    x_train, x_test = mnist_prepare()  # pre-process mnist
    x_train, x_test = scale_data(x_train, x_test)
    # l_number_of_samples = np.linspace(250, 900, 4, endpoint=False)
    l_number_of_samples = np.concatenate(
        (np.linspace(250, 900, 4, endpoint=False), np.linspace(900, 10000, 20, endpoint=True),
         np.linspace(10001, 20000, 5, endpoint=True),
         np.linspace(20001, 40000, 4, endpoint=True)), axis=None)
    np_q = list_q(l_number_of_samples, delta)  # calculate size of q
    data_to_csv = np.column_stack((l_number_of_samples, np_q))
    plt.subplot(1, 3, 1)
    plt.title('MNIST: SKM Vs offline - delta:%s,iter: %s' % (delta, number_of_iteration_per_m))
    plt.xlabel('number of samples')
    plt.ylabel('risk')
    for i, k in enumerate([10, 15, 20]):  #
        np_rate, np_offline_risk = exp2(x_train, x_test, k, delta, number_of_iteration_per_m,
                                        l_number_of_samples, birch_kmedoids_mode)  # calculate rate and offline risk
        data_to_csv = np.column_stack((data_to_csv, np_rate))
        data_to_csv = np.column_stack((data_to_csv, np_offline_risk))
        plt.plot(l_number_of_samples, np_rate, marker[i], label='k=%s' % k)
    to_csv(data_to_csv)
    plt.legend(loc='upper right')


def exp_census(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker):
    x_train, x_test = census1990_prepare()
    x_train, x_test = scale_data(x_train, x_test)
    l_number_of_samples = np.concatenate((np.linspace(250, 5000, 10, endpoint=True),
                                          np.linspace(5001, 18575, 13, endpoint=True)), axis=None)
    np_q = list_q(l_number_of_samples, delta)
    data_to_csv = np.column_stack((l_number_of_samples, np_q))
    plt.subplot(1, 3, 2)
    plt.title('census- SKM versus offline - delta:%s ,iter: %s' % (delta, number_of_iteration_per_m))
    plt.xlabel('number of samples')
    plt.ylabel('risk')
    for i, k in enumerate([5, 10, 15, 20]):  #
        np_rate, np_offline_risk = exp2(x_train, x_test, k, delta, number_of_iteration_per_m,
                                        l_number_of_samples, birch_kmedoids_mode)  # calculate rate and offline risk
        data_to_csv = np.column_stack((data_to_csv, np_rate))
        data_to_csv = np.column_stack((data_to_csv, np_offline_risk))
        plt.plot(l_number_of_samples, np_rate, marker[i], label='k=%s' % k)
    to_csv(data_to_csv)
    plt.legend(loc='upper right')


def exp_forest(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker):
    x_train, x_test = forest_data_prepare()
    x_train, x_test = scale_data(x_train, x_test)
    print(x_train.shape[0], x_train.shape[1], x_test.shape[0])
    l_number_of_samples = np.concatenate((np.linspace(250, 5000, 10, endpoint=True),
                                          np.linspace(5001, 20000, 5, endpoint=True),
                                          np.linspace(20001, 57999, 4, endpoint=True)), axis=None)
    np_q = list_q(l_number_of_samples, delta)
    data_to_csv = np.column_stack((l_number_of_samples, np_q))
    plt.subplot(1, 3, 3)
    plt.title('covtype- SKM versus offline - delta:%s ,iter: %s' % (delta, number_of_iteration_per_m))
    plt.xlabel('number of samples')
    plt.ylabel('risk')
    for i, k in enumerate([5, 10, 20]):  #
        np_rate, np_offline_risk = exp2(x_train, x_test, k, delta, number_of_iteration_per_m,
                                        l_number_of_samples, birch_kmedoids_mode)  # calculate rate and offline risk
        data_to_csv = np.column_stack((data_to_csv, np_rate))
        data_to_csv = np.column_stack((data_to_csv, np_offline_risk))
        plt.plot(l_number_of_samples, np_rate, marker[i], label='k=%s' % k)
    to_csv(data_to_csv)
    plt.legend(loc='upper right')


def main_experiment(delta, number_of_iteration_per_m, birch_kmedoids_mode):
    """
       Running  experiments on 3 data sets, MNIST, census and covertype. Calculate the risk ratio between SKM and offline algorithm
       :return: Graph and output 3 csv files.
    """
    plt.subplots(1, 3, figsize=(15, 60))
    marker = ['b--', 'k--', 'r--', 'm--']
    # mnist
    exp_mnist(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker)
    # census data set
    exp_census(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker)
    # forest data set
    exp_forest(delta, number_of_iteration_per_m, birch_kmedoids_mode, marker)
    # show plot
    plt.show()


def main():
    main_experiment(0.01, 20, 1)
    return


if __name__ == '__main__':
    main()
