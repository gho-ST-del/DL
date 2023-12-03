import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像


# 计算欧式距离
def euclidean_distance(hist1, hist2):
    return np.sqrt(np.sum((hist1 - hist2) ** 2))
def calculate_histogram_distances(h_bins):
    s_bins=h_bins
    # 读取图像
    rgb1 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')
    rgb2 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12_balanced.png')
    gray1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    hsv1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2HSV)

    # 计算 HSV 直方图
    hist_size = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    hist1_hsv = cv2.calcHist([hsv1], [0, 1], None, hist_size, ranges, accumulate=False)
    hist2_hsv = cv2.calcHist([hsv2], [0, 1], None, hist_size, ranges, accumulate=False)

    # 计算 RGB 直方图
    hist_size = [256]
    ranges = [0, 256]
    hist1_rgb = cv2.calcHist([rgb1], [0], None, hist_size, ranges, accumulate=False) + \
                cv2.calcHist([rgb1], [1], None, hist_size, ranges, accumulate=False) + \
                cv2.calcHist([rgb1], [2], None, hist_size, ranges, accumulate=False)
    hist2_rgb = cv2.calcHist([rgb2], [0], None, hist_size, ranges, accumulate=False) + \
                cv2.calcHist([rgb2], [1], None, hist_size, ranges, accumulate=False) + \
                cv2.calcHist([rgb2], [2], None, hist_size, ranges, accumulate=False)

    # 计算灰度直方图
    hist_size = [64]
    ranges = [0, 256]
    hist1_gray = cv2.calcHist([gray1], [0], None, hist_size, ranges, accumulate=False)
    hist2_gray = cv2.calcHist([gray2], [0], None, hist_size, ranges, accumulate=False)

    # 归一化直方图
    cv2.normalize(hist1_hsv, hist1_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2_hsv, hist2_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist1_rgb, hist1_rgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2_rgb, hist2_rgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist1_gray, hist1_gray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2_gray, hist2_gray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
 
 
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
    # 绘制直方图
    axs[0, 0].imshow(rgb1)
    axs[0, 0].set_title('Image 1')
    axs[0, 1].imshow(rgb2)
    axs[0, 1].set_title('Image 2')
    axs[0, 2].hist(hist1_rgb)
    axs[0, 2].set_title('RGB Histogram 1')
    axs[0, 3].hist(hist2_rgb)
    axs[0, 3].set_title('RGB Histogram 2')
    axs[1, 0].hist(hist1_gray)
    axs[1, 0].set_title('Gray Histogram 1')
    axs[1, 1].hist(hist2_gray)
    axs[1, 1].set_title('Gray Histogram 2')
    axs[1, 2].hist(hist1_hsv)
    axs[1, 2].set_title('HSV Histogram 1')
    axs[1, 3].hist(hist2_hsv)
    axs[1, 3].set_title('HSV Histogram 2')
    plt.show()

    # 计算欧式距离
    dist_hsv = euclidean_distance(hist1_hsv, hist2_hsv)
    dist_rgb = euclidean_distance(hist1_rgb, hist2_rgb)
    dist_gray = euclidean_distance(hist1_gray, hist2_gray)

    # 显示结果
    print('HSV distance:', dist_hsv)
    print('RGB distance:', dist_rgb)
    print('Gray distance:', dist_gray)

    return dist_hsv, dist_rgb, dist_gray

calculate_histogram_distances(24)
calculate_histogram_distances(32)
calculate_histogram_distances(64)








