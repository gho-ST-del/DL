from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
data = iris.data
labels = iris.target

scaler = StandardScaler()
data = scaler.fit_transform(data)

# 计算相异矩阵
dissimilarity_matrix = pairwise_distances(data, metric='manhattan')

# 层次聚类，尝试使用"ward"链接方法
linkage_matrix = linkage(dissimilarity_matrix, method='complete')

# 使用fcluster函数进行截断，得到每个样本的聚类标签，尝试使用不同的截断策略
cluster_labels = fcluster(linkage_matrix,t=3.0, criterion='maxclust')

acc = np.mean(labels+1 == cluster_labels)
print(acc)

# 绘制层次聚类树状图
dendrogram(linkage_matrix, labels=range(len(data)), orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()