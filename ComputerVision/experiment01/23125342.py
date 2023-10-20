# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # 用于处理数组和矩阵的Python库
from skimage.feature import local_binary_pattern  # 用于提取局部二进制模式特征的函数
from sklearn.cluster import KMeans  # 用于实现KMeans聚类算法的类
import matplotlib.pyplot as plt  # 用于绘制图表的Python库

# 定义KMeans函数
def f_kmeans(cluster=50, n=50, pixels=None):
    # 创建KMeans对象
    kmeans = KMeans(n_clusters=cluster, random_state=0, n_init=50)
    return kmeans.fit(pixels)

# 定义显示图像的函数
def show_img(kmeans, cluster, name):
    # 获取每个像素的标签
    labels = kmeans.labels_

    # 将标签转换回图像尺寸
    height, width, _ = image.shape
    segmented_image = np.reshape(labels, (height, width)).astype(np.uint8)

    # 合并所有类别的像素到一张图像中
    merged_image = np.zeros_like(image)
    for i in range(cluster):
        mask = cv2.inRange(segmented_image, i, i)
        color = np.random.randint(0, 255, size=3)
        color_image = np.zeros_like(image)
        color_image[mask > 0] = color
        merged_image = cv2.add(merged_image, color_image)

    # 显示合并后的图像
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.imwrite(name, merged_image)

# 读取图像
image = cv2.imread(r'E:\PythonCode\DeepLearning\ComputerVision\data\data01\twins.jpg')

# 将图像转换为灰度空间
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 提取 LBP 特征
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
lbp_vector = lbp_image.reshape(-1, 1)

# 将图像转换为 RGB 空间
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像转换为 HSV 空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 将图像转换为一维数组
rgb_pixels = rgb_image.reshape((-1, 3))
hsv_pixels = hsv_image.reshape((-1, 3))

# 定义聚类数目
CLU = 20
hsv_vector = hsv_image.reshape(-1, 3)
features = np.concatenate((hsv_vector, lbp_vector), axis=1)
concat_feature = f_kmeans(cluster=CLU, n=50, pixels=features)
show_img(concat_feature, CLU, 'concat.jpg')

# 运行 K-Means 聚类算法
CLU = 50
hsv_kmeans = f_kmeans(cluster=CLU, n=50, pixels=hsv_pixels)
rgb_kmeans = f_kmeans(cluster=CLU, n=50, pixels=rgb_pixels)
show_img(rgb_kmeans, CLU, 'rgb.jpg')
show_img(hsv_kmeans, CLU, 'hsv.jpg')

CLU = 8
lbp_kmeans = f_kmeans(cluster=CLU, n=50, pixels=lbp_vector)
show_img(lbp_kmeans, CLU, 'lbp.jpg')
