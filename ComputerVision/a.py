import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 定义kmeans函数
def kmeans_cluster(cluster=50, n=50, pixels=None):
    kmeans = KMeans(n_clusters=cluster, random_state=0, n_init=50)
    return kmeans.fit(pixels)

# 定义显示图像的函数
def show_image(kmeans, cluster, name, image):
    labels = kmeans.labels_
    height, width, _ = image.shape
    segmented_image = np.reshape(labels, (height, width)).astype(np.uint8)
    merged_image = np.zeros_like(image)
    for i in range(cluster):
        mask = cv2.inRange(segmented_image, i, i)
        color = np.random.randint(0, 255, size=3)
        color_image = np.zeros_like(image)
        color_image[mask > 0] = color
        merged_image = cv2.add(merged_image, color_image)
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.imwrite(name, merged_image)

# 读取图像
image = cv2.imread(r'E:\PythonCode\DeepLearning\ComputerVision\data\data01\twins.jpg')

# 将图像转换为灰度空间，并提取LBP特征
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
lbp_vector = lbp_image.reshape(-1, 1)

# 将图像转换为RGB和HSV空间，并将它们转换为一维数组
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
rgb_pixels = rgb_image.reshape((-1, 3))
hsv_pixels = hsv_image.reshape((-1, 3))

# 运行KMeans聚类算法，并显示合并后的图像
cluster = 20
hsv_vector = hsv_image.reshape(-1, 3)
features = np.concatenate((hsv_vector, lbp_vector), axis=1)
concat_feature = kmeans_cluster(cluster=cluster, n=50, pixels=features)
show_image(concat_feature, cluster, 'concat.jpg', image)

cluster = 50
hsv_kmeans = kmeans_cluster(cluster=cluster, n=50, pixels=hsv_pixels)
rgb_kmeans = kmeans_cluster(cluster=cluster, n=50, pixels=rgb_pixels)
show_image(rgb_kmeans, cluster, 'rgb.jpg', image)
show_image(hsv_kmeans, cluster, 'hsv.jpg', image)

cluster = 8
lbp_kmeans = kmeans_cluster(cluster=cluster, n=50, pixels=lbp_vector)
show_image(lbp_kmeans, cluster, 'lbp.jpg', image)