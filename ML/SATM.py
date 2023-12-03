import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_kernels
from scipy.sparse.linalg import eigs
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
data = iris.data
labels = iris.target



def slonim_satb_clustering(data, num_clusters, alpha=0.1, beta=0.1, num_eigenvectors=None):
    # 1. 计算谱矩阵
    similarity_matrix = pairwise_kernels(data, metric='rbf', gamma=1.0)
    diagonal_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = diagonal_matrix - similarity_matrix

    # 2. 计算特征向量
    if num_eigenvectors is None:
        num_eigenvectors = num_clusters
    _, eigenvectors = eigs(laplacian_matrix, k=num_eigenvectors, which='SM')

    # 3. 构建扩展矩阵，考虑时间偏差
    extended_matrix = np.hstack([eigenvectors.real, beta * np.arange(1, len(data) + 1).reshape(-1, 1)])

    # 4. K均值聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(extended_matrix)

    return labels

# 示例用法

num_clusters = 3

result_labels = slonim_satb_clustering(data, num_clusters)
print("Cluster Labels:", result_labels)
print("Accuracy:", np.mean(labels == result_labels))