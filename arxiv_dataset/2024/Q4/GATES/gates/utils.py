import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata, save=False, plt_show=True):
    import matplotlib.pyplot as plt

    # 计算网络的边数和平均边数
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]

    # 统计每个节点的邻居数量分布
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    if plt_show:

        # 创建图形
        fig, ax = plt.subplots(figsize=[3, 2])
        plt.ylabel('Percentage')
        plt.xlabel('')
        plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)

        # 绘制柱状图
        ax.bar(plot_df.index, plot_df)

        # 保存图像
        if save:
            plt.tight_layout()  # 确保布局不重叠
            plt.savefig(save, dpi=300, bbox_inches='tight')  # 保存为指定文件
        # 显示图像
        plt.show()


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=True):
    """
    计算细胞间基因表达的相似度，并构建稀疏矩阵。

    Parameters:
    -----------
    adata : AnnData
        包含基因表达数据的 AnnData 对象
    k_neighbors : int
        每个细胞选择最相似的邻居数量
    metric : str
        相似度度量方式，可选 'cosine', 'euclidean', 'correlation', 'pearson'

    Returns:
    --------
    KNN_df : pd.DataFrame
        稀疏矩阵，包含列 ['Cell1', 'Cell2', 'Distance']
    """

    # 将基因表达矩阵转为DataFrame，并设定细胞的index
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    # adata_Vars = adata
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)

    # 根据选择的度量方法计算相似度
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(X)
    elif metric == 'euclidean':
        similarity_matrix = -euclidean_distances(X)  # 将距离转为相似度，距离越小相似度越大
    elif metric == 'pearson':
        # 使用皮尔逊相关系数
        similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                similarity_matrix[i, j] = pearsonr(X.iloc[i, :], X.iloc[j, :])[0]
    else:
        raise ValueError(f"未知的相似度度量: {metric}")

    # 存储最相似的n_neighbors个细胞的关系
    KNN_list = []

    for i in range(similarity_matrix.shape[0]):
        # 获取与当前细胞最相似的n_neighbors个邻居（排除自身）
        sorted_indices = np.argsort(-similarity_matrix[i, :])  # 按相似度降序排序
        closest_cells = sorted_indices[1:k_neighbors + 1]  # 取最相似的n_neighbors个细胞（排除自己）
        closest_distances = similarity_matrix[i, closest_cells]  # 对应的相似度

        # 将每个细胞与其最相似的细胞对存入KNN_list
        KNN_list.append(pd.DataFrame({
            'Cell1': [i] * k_neighbors,
            'Cell2': closest_cells,
            'Distance': closest_distances
        }))

    # 将所有DataFrame合并
    KNN_df = pd.concat(KNN_list, ignore_index=True)

    # 将细胞编号映射回原来的index
    id_cell_trans = dict(zip(range(X.shape[0]), X.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (KNN_df.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (KNN_df.shape[0] / adata.n_obs))

    # 保存到 adata
    adata.uns['Gene_Similarity_Net'] = KNN_df