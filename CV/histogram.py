import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')
img2 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12_balanced.png')

# 将图像转换为 HSV、RGB 和灰度空间
hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 计算 HSV 直方图
h_bins = 24
s_bins = 32
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

# 绘制直方图
plt.subplot(2, 2, 1)
plt.imshow(rgb1)
plt.title('Image 1')
plt.subplot(2, 2, 2)
plt.imshow(rgb2)
plt.title('Image 2')
plt.subplot(2, 2, 3)
plt.hist(hist1_rgb)
plt.title('RGB Histogram 1')
plt.subplot(2, 2, 4)
plt.hist(hist2_rgb)
plt.title('RGB Histogram 2')
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(gray1, cmap='gray')
plt.title('Image 1')
plt.subplot(2, 2, 2)
plt.imshow(gray2, cmap='gray')
plt.title('Image 2')
plt.subplot(2, 2, 3)
plt.hist(hist1_gray)
plt.title('Gray Histogram 1')
plt.subplot(2, 2, 4)
plt.hist(hist2_gray)
plt.title('Gray Histogram 2')
plt.show()
# 计算欧式距离
dist_hsv = cv2.compareHist(hist1_hsv, hist2_hsv, cv2.HISTCMP_BHATTACHARYYA)
dist_rgb = cv2.compareHist(hist1_rgb, hist2_rgb, cv2.HISTCMP_BHATTACHARYYA)
dist_gray = cv2.compareHist(hist1_gray, hist2_gray, cv2.HISTCMP_BHATTACHARYYA)

# 显示结果
print('HSV distance:', dist_hsv)
print('RGB distance:', dist_rgb)
print('Gray distance:', dist_gray)