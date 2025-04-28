import numpy as np
import geatpy as ea
from Initialization import Individual,Individual1
import networkx as nx
import random
import math
from itertools import combinations

def get_upper_triangular_vector(matrix):
    """
    获取邻接矩阵的上三角部分，转换为一维向量
    """
    return [matrix[i, j] for i in range(matrix.shape[0]) for j in range(i + 1, matrix.shape[1])]

def vector_to_matrix(vector, size):
    """
    将上三角向量转换为邻接矩阵
    """
    matrix = np.zeros((size, size), dtype=int)
    upper_tri_indices = list(combinations(range(size), 2))
    for index, (i, j) in enumerate(upper_tri_indices):
        matrix[i, j] = vector[index]
    return matrix + matrix.T  # 对称矩阵

def graph_hash(graph):
    """
    生成图的哈希值，用于比较图是否同构
    """
    return nx.weisfeiler_lehman_graph_hash(graph, iterations=3, digest_size=16)


def remove_duplicate_graphs(vectors, size):
    """
    移除具有相同图结构的向量
    """
    unique_graphs = {}
    for vector in vectors:
        matrix = vector_to_matrix(vector, size)
        graph = nx.from_numpy_matrix(matrix)
        graph_hash_value = graph_hash(graph)
        if graph_hash_value not in unique_graphs:
            unique_graphs[graph_hash_value] = vector
    return list(unique_graphs.values())

class CustomGraph:
    def __init__(self, l, k):
        self.l = l
        self.k = k
        self.graph = nx.Graph()
        self.create_graph()

    def create_graph(self):
        # 第一部分：有 (l-1) 个顶点的完全图
        complete_graph1 = nx.complete_graph(self.l - 1)
        self.graph = nx.compose(self.graph, complete_graph1)

        # 第二部分：l 个有 k 个顶点的完全图
        for i in range(self.l):
            complete_graph2 = nx.complete_graph(self.k)
            # Shift node labels for the second part
            offset = (self.l - 1) + i * self.k
            mapping = {node: node + offset for node in complete_graph2.nodes()}
            complete_graph2 = nx.relabel_nodes(complete_graph2, mapping)
            self.graph = nx.compose(self.graph, complete_graph2)

            # 连接第一部分的每个点到第二部分的每个点
            for node1 in range(self.l - 1):
                for node2 in complete_graph2.nodes():
                    self.graph.add_edge(node1, node2)

    def get_upper_triangular(self):
        adj_matrix = nx.to_numpy_array(self.graph)
        upper_triangular = adj_matrix[np.triu_indices(len(self.graph), k=1)]
        return upper_triangular



def is_integer(value):
    return isinstance(value, int)


def Com_kt(n):
    k = 2
    K = k

    if n>= 4*k-5:
        for i in range(k,int(n/2)):
            l = (n+1)/(K+1)
            if is_integer(l):  # 如果l是整数 则通过 l 和 k 来构建图
                custom_graph = CustomGraph(l, k)
                upper_triangular = custom_graph.get_upper_triangular()
            K = K+1
    else:
        pass

def ComDeg(code,n):  # 计算每个顶点的度
    Deg = np.zeros(n)

    # 初始化5x5的邻接矩阵
    adj_matrix = np.zeros((n, n), dtype=int)

    # 将向量的值填充到邻接矩阵的上三角部分
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj_matrix[i, j] = code[k]
            adj_matrix[j, i] = code[k]
            k += 1
    degrees = np.sum(adj_matrix, axis=1)
    for i in range(len(Deg)):
        Deg[i] = degrees[i]
    return Deg

def index_upper_triangle(n, i, j):
    """计算上三角矩阵的线性索引（不包括对角线）"""
    if i >= j:
        raise ValueError("i must be less than j for upper triangle matrix")
    return int(i * n - i * (i + 1) // 2 + j - i - 1)


def delete_node(vec, n, k):
    """删除节点 k 并更新上三角矩阵向量"""
    new_vec = []
    for i in range(n):
        for j in range(i + 1, n):
            if i == k or j == k:
                continue
            new_vec.append(vec[index_upper_triangle(n, i, j)])

    # 返回更新后的向量和新的节点总数
    return new_vec, n - 1

def Com_IG(pop,n):
    delt = 1 # 每次选择delt个点。
    new_G = pop
    # 选择要删的点
    new_n = n
    S = 0 # 扣掉的顶点数
    IG = 9999 # 初始孤立韧度
    new_Deg = ComDeg(new_G, new_n)
    for j in range(n-1):
        for d in range(delt):
            r_n = random.random()   # 随机生成一个0-1的数
            for i in range(new_n):
                if (np.sum(new_Deg[:i]))/(np.sum(new_Deg[:])) <= r_n < (np.sum(new_Deg[:i+1]))/(np.sum(new_Deg[:])):
                    # 轮盘赌判断
                    I = int(i) # 要删除的点，-1是因为从0开始
                    S= S+1
                    break
            #构建新的图，把顶点I删掉
            new_G,new_n = delete_node(new_G,new_n,I)   # 删除第I个顶点后的图,
            # 重新计算新的图结构G-S中的孤立点数量，也就是有几个点的度为0。
            new_Deg = ComDeg(new_G, new_n)
            if count_zeros(new_Deg)>=2:
                if IG > S/((count_zeros(new_Deg))-1):
                    IG = S/((count_zeros(new_Deg))-1)

    return IG

def count_zeros(vector):
    return np.count_nonzero(np.array(vector) == 0)


def single_point_crossover(parent1, parent2):
    """
    单点交叉函数
    参数:
    parent1, parent2: 需要交叉的两个父代个体（数组）

    返回:
    offspring1, offspring2: 生成的两个子代个体（数组）
    """
    # 确定交叉点
    crossover_point = np.random.randint(1, len(parent1))

    # 生成子代
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return offspring1, offspring2


def binary_mutation(offspring, mutation_rate=0.3):
    """
    二进制变异函数
    参数:
    offspring: 需要变异的子代个体（数组）
    mutation_rate: 变异概率

    返回:
    mutated_offspring: 变异后的子代个体（数组）
    """
    # 生成一个与子代个体等长的随机数数组
    random_values = np.random.random(len(offspring))

    # 对于每个基因，按变异概率进行变异
    for i in range(len(offspring)):
        if random_values[i] < mutation_rate:
            offspring[i] = 1 - offspring[i]  # 翻转位

    return offspring


def shuffle_and_pair(population):
    """
    打乱种群并成对选择父代
    参数:
    population: 种群（数组的数组）

    返回:
    pairs: 成对的父代个体列表（数组的数组）
    """
    np.random.shuffle(population)
    pairs = [(population[i], population[i + 1]) for i in range(0, len(population), 2)]
    return pairs


def add_vector_to_element(lst, index, new_vector):
    """
    向列表中指定索引的元组添加一个新的向量

    参数:
    lst (list): 包含元组的列表
    index (int): 需要添加新向量的元组的索引
    new_vector (list): 要添加的新向量
    """
    if index < 0 or index >= len(lst):
        raise IndexError("索引超出列表范围")
    # 获取指定索引的元组
    element = lst[index]
    # 创建一个新的元组，包括原始的元素和新的向量
    new_element = element + (new_vector,)
    # 替换原始列表中的元素
    lst[index] = new_element



def remove_duplicates(lst):
    """
    移除列表中元组的相同副本，只保留第一个向量。

    参数:
    lst (list): 包含元组的列表
    """
    for i in range(len(lst)):
        element = lst[i]
        unique_vectors = []
        seen_vectors = set()
        for sublist in element[1:]:
            # 将子列表转换为元组以便于集合操作
            sublist_tuple = tuple(sublist)
            if sublist_tuple not in seen_vectors:
                seen_vectors.add(sublist_tuple)
                unique_vectors.append(sublist)
        # 保留第一个元素并替换后面的向量
        lst[i] = (element[0],) + tuple(unique_vectors)


def hamming_distance(vec1, vec2):
    return np.sum(vec1 != vec2)

def find_vectors(vectors, m):
    selected_vectors = []

    # Convert list of arrays to numpy array
    vectors = np.array(vectors)

    # Step 1: Select the vector with the most 0s
    zero_counts = np.sum(vectors == 0, axis=1)
    first_vector_index = np.argmax(zero_counts)
    selected_vectors.append(vectors[first_vector_index])
    remaining_vectors = np.delete(vectors, first_vector_index, axis=0)

    # Step 2 & 3: Iteratively select the vector with the maximum sum of Hamming distances
    for _ in range(m - 1):
        max_distance_sum = -1
        selected_index = -1

        for i, vec in enumerate(remaining_vectors):
            distance_sum = sum(hamming_distance(vec, sel_vec) for sel_vec in selected_vectors)/(len(selected_vectors))

            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                selected_index = i

        selected_vectors.append(remaining_vectors[selected_index])
        remaining_vectors = np.delete(remaining_vectors, selected_index, axis=0)

    return selected_vectors

N= 10 #种群数量
r_1=0.5
k = 4 # 分数k因子
n=15 # 顶点数量
# k_min = 2 # 最小度（k因子）
TMax = 100 # 最大迭代次数
# population = np.asarray([Individual(n=n)  # 初始化个体
#                          for _ in range(N)])
D = int((n*(n-1))/2)
# I_pop = []  # 反例种群
# T = 0
K = k  # 反例中的k
# 生成反例，可能找不到，找到后通过突变生成指定数量的反例种群。
num = {}
IGnum = []
Round = 10
round = 0
while(round<Round):
    I_pop = []  # 反例种群
    T = 0
    if n >= 4 * k - 5:
        for i in range(k, int(math.ceil(n/2))):
            l = (n + 1) / (K + 1)
            if is_integer(l):  # 如果l是整数 则通过 l 和 k 来构建图
                custom_graph = CustomGraph(l, k)  # 构图
                upper_triangular = custom_graph.get_upper_triangular() # 保存上三角矩阵 （编码）
                I_pop.append(upper_triangular)  # 保存编码
            K = K + 1
        setlenth = int(math.ceil(n/2) - k)
    else:
        for i in range(k, n):
            l = (n + 1) / (K + 1)
            if is_integer(l):  # 如果l是整数 则通过 l 和 k 来构建图
                custom_graph = CustomGraph(l, k)
                upper_triangular = custom_graph.get_upper_triangular()
                I_pop.append(upper_triangular)
            K = K + 1
        setlenth = int(math.ceil(n) - k)
    optset = [(9999, [0 for j in range(0, D)]) for i in range(setlenth)]  # 最优解集,setlenth是k能取的数量
    population= [] # 种群
    # 初始化过程
    if 0<len(I_pop)<N*r_1 :   # 反例种群突变
        I_n = int(N * r_1 - len(I_pop))
        for i in range(I_n):
            I_pop.append(binary_mutation(I_pop[i]))
        for i in range(N - int(N*r_1)):
            population.append(np.reshape(np.random.randint(2, size=D), -1))   # 随机初始化

    elif len(I_pop)==0:
        for i in range(N):
            population.append(np.reshape(np.random.randint(2, size=D), -1))   # 随机初始化
    population = population + I_pop

    # 将列表转换为 NumPy 数组


    while(T<=TMax):


        # 评估， 判断每个个体是否满足孤立韧度变种要求，和具体的孤立韧度变种数值
        for i in range(len(population)):
            IG = Com_IG(population[i],n)  # 计算孤立韧度
            # 计算最小度 mink
            minK = np.min(ComDeg(population[i],n))
            t = int(minK-k)
            if setlenth - k >= t >= 0:
                if IG > k+((k-1)/(t+1)):    # 判断是否满足孤立韧度和最小度是否达到条件的条件
                    if optset[t][0] > IG:
                        optset[t:t+1]= [(IG,population[i])]
                    elif optset[t][0] == IG:
                        add_vector_to_element(optset, t, population[i])
            # print(IG)
        ##交叉变异 单点交叉、高斯变异
        pairs = shuffle_and_pair(population)
        Q = [] # 下一个种群
        for parent1, parent2 in pairs:
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            offspring1 = binary_mutation(offspring1)
            offspring2 = binary_mutation(offspring2)
            Q.append(offspring1)
            Q.append(offspring2)
        population = Q

        T = T+1

    remove_duplicates(optset)
    # print(optset)
    for i in range(len(optset)):
        optset[i] = (optset[i][0],remove_duplicate_graphs(optset[i][1:], n))

        if round==0:
            num[i+k] = [len(optset[i][1][:])]
            IGnum.append(optset[i][0])
        else:
            num[i+k].append(len(optset[i][1][:]))
            if optset[i][0]< IGnum[i]:
                IGnum[i] = optset[i][0]
        # print(len(optset[i][1])-1)
        if len(list(optset[i][1][:]))>N:
            selected_vectors = find_vectors(list(optset[i][1][:]), N)
            # print(optset[i][0],":Selected vectors:\n", selected_vectors)
            # print('----------')
        else:
            pass
            # print(optset[i])
            # print('----------')
    round = round+1
# J = 0
for key in num:
    print(np.mean(num[key]))
    print(np.std(num[key]))

# print(num)
# print(IGnum)
