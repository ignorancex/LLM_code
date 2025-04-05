# encoding=utf-8
"""
    Created on 14:52 2017/4/30 
    @author: Jindong Wang
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd

class TCA:
    #dim = 5
    #kerneltype = 'rbf'
    #kernelparam = 1
    #mu = 1

    def __init__(self, dim=5, kerneltype='rbf', kernelparam=1, mu=1):
        '''
        Init function
        :param dim: dims after tca (dim <= d)
        :param kerneltype: 'rbf' | 'linear' | 'poly' (default is 'rbf')
        :param kernelparam: kernel param
        :param mu: param
        '''
        self.dim = dim
        self.kernelparam = kernelparam
        self.kerneltype = kerneltype
        self.mu = mu

    def get_L(self, n_src, n_tar):
        '''
        Get index matrix
        :param n_src: num of source domain 
        :param n_tar: num of target domain
        :return: index matrix L
        '''
        """
        np.full -> 最初のタプルで指定されたshapeのarrayを，2番目の引数で指定された値で埋める．
        以下の例では，最初のタプロで指定された大きさの配列に1が埋められる．
        つまりその前方の式と合わせると，指定された大きさの配列で，最初の要素で計算されている値が埋められる．
        それぞれは，入力されたファイル数と同等の大きさの行と列を持った行列になる．
        pan2008AAAIの論文を見れば，これがあっていることがわかる．
        """
        L_ss = (1. / (n_src * n_src)) * np.full((n_src, n_src), 1)
        L_st = (-1. / (n_src * n_tar)) * np.full((n_src, n_tar), 1)
        L_ts = (-1. / (n_tar * n_src)) * np.full((n_tar, n_src), 1)
        L_tt = (1. / (n_tar * n_tar)) * np.full((n_tar, n_tar), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        return L

    def get_kernel(self, kerneltype, kernelparam, x1, x2=None):
        '''
        Calculate kernel for TCA (inline func)
        :param kerneltype: 'rbf' | 'linear' | 'poly'
        :param kernelparam: param
        :param x1: x1 matrix (n1,d)
        :param x2: x2 matrix (n2,d)
        :return: Kernel K
        '''
        n1, dim = x1.shape
        K = None
        if x2 is not None:
            n2 = x2.shape[0]
        if kerneltype == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif kerneltype == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1, x2.T), kernelparam)
            else:
                K = np.power(np.dot(x1, x1.T), kernelparam)
        elif kerneltype == 'rbf':
            if x2 is not None:
                sum_x2 = np.sum(np.multiply(x2, x2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (
                    np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) + np.tile(sum_x2, (1, n1)) - 2 * np.dot(x2,
                                                                                                                    x1.T)) / (
                               dim * 2 * kernelparam))
            else:
                #P = np.sum(np.multiply(x1, x1), axis=1)
                #P = P.reshape((len(P), 1)) # 第二引数が1なので，1列にする．
                #K = np.exp(
                #    -1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * kernelparam))
                """
                上が元々の実装である．
                ただ，内容がよくわからなかったので，scikit-learnで置換した．kernelparamは握りつぶしている．
                """
                K = rbf_kernel(x1)

        return K

    def fit_transform(self, x_src, x_tar, x_tar_o=None):
        '''
        TCA main method. Wrapped from Sinno J. Pan and Qiang Yang's "Domain adaptation via transfer component ayalysis. IEEE TNN 2011" 
        :param x_src: Source domain data feature matrix. Shape is (n_src,d)
        :param x_tar: Target domain data feature matrix. Shape is (n_tar,d)
        :param x_tar_o: Out-of-sample target data feature matrix. Shape is (n_tar_o,d)
        :return: tranformed x_src_tca,x_tar_tca,x_tar_o_tca
        '''
        n_src = x_src.shape[0]
        n_tar = x_tar.shape[0]
        X = np.vstack((x_src, x_tar)) # combine x_src and x_tar along y-axis.
        L = self.get_L(n_src, n_tar) 
        L[np.isnan(L)] = 0 # replace NaN to 0 in L. np.isnan(L) return boolean matrix. If original value is NaN, True, otherwise False.
        K = self.get_kernel(self.kerneltype, self.kernelparam, X)
        K[np.isnan(K)] = 0 # replace NaN to 0 in K.
        if x_tar_o is not None:
            K_tar_o = self.get_kernel(self.kerneltype, self.kernelparam, X, x_tar_o)

        """
        Calculating H. identity -> creating 単位行列. 
        """
        H = np.identity(n_src + n_tar) - 1. / (n_src + n_tar) * np.ones(shape=(n_src + n_tar, 1)) * np.ones(
            shape=(n_src + n_tar, 1)).T

        """
        この下のmuの位置が逆では？
        と思ったけど，ジャーナルペーパー (matasci2011TCAJounal) ではこれであっている記述になっている．
        論文の実装を間違えたのか？
        """
        forPinv = self.mu * np.identity(n_src + n_tar) + np.dot(np.dot(K, L), K)
        forPinv[np.isnan(forPinv)] = 0

        Kc = np.dot(np.dot(np.dot(np.linalg.pinv(forPinv), K), H), K)
        Kc[np.isnan(Kc)] = 0

        """
        Above here, they calculate optimization of equation (8).
        Below here, they calculate eigenvalues of (I + \mu{}KLK)^-1NHK.
        These eigenvalues are components of W.
        """

        D, V = np.linalg.eig(Kc) # returns the eigenvalues and the normalized eighenvectors. Each vectors (e.g. V[:,i]) are corresponding to the eigenvalue (D[i])
        eig_values = D.reshape(len(D), 1)
        #eig_values_sorted = np.sort(eig_values[::-1], axis=0)
        #print(eig_values.shape)
        #print(V.shape)
        index_sorted = np.argsort(-eig_values, axis=0) # eig_valuesは一列のeigenvalueが格納されている．これをソートする．
        V = V[:, index_sorted] # 上でソートされた値を元に，eigenvectorsをソートする．各列が一つのeigenvectorなので，インデックスを指定している
        V = V.reshape((V.shape[0], V.shape[1])) # 上の状態だと，arrayが3次元になっているので，2次元に直してあげる．これにより，eigenvalueの値によって，eigenvectorsの列の位置が変更される．


        x_src_tca = np.dot(K[:n_src, :], V)
        x_tar_tca = np.dot(K[n_src:, :], V)
        if x_tar_o is not None:
            x_tar_o_tca = np.dot(K_tar_o, V)
        else:
            x_tar_o_tca = None

        """
        asarrayもarrayと同じくnp.arrayに変更するものである．
        """
        x_src_tca = np.asarray(x_src_tca[:, :self.dim], dtype=float)
        x_tar_tca = np.asarray(x_tar_tca[:, :self.dim], dtype=float)
        if x_tar_o is not None:
            x_tar_o_tca = x_tar_o_tca[:, :self.dim]



        return x_src_tca, x_tar_tca, x_tar_o_tca


if __name__ == '__main__':
    file_path = 'data/test_tca_data.csv'
    data = np.loadtxt(file_path, delimiter=',')
    x_src = data[:, :81]
    x_tar = data[:, 81:]

    # example usage
    my_tca = TCA(dim=30)
    x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(x_src, x_tar)
    np.savetxt('x_src1.csv', x_src_tca, delimiter=',', fmt='%.6f')
    np.savetxt('x_tar1.csv', x_tar_tca, delimiter=',', fmt='%.6f')


