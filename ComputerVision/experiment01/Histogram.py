import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image1 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12_balanced.png')
image2 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')

# 转换图像到不同的颜色空间
image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# 计算直方图
bins = [24, 32, 64]
histograms = []

for num_bins in bins:
    for color_space, image in [("RGB", image1), ("HSV", image1_hsv)]:
        if color_space == "RGB":
            hist = cv2.calcHist([image], [0, 1, 2], None, [num_bins, num_bins, num_bins], [0, 256, 0, 256, 0, 256])
            # print(hist)
        elif color_space == "HSV":
            hist = cv2.calcHist([image], [0, 1], None, [num_bins, num_bins], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append((color_space, num_bins, hist))

# 计算灰度直方图
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

for num_bins in bins:
    hist = cv2.calcHist([image1_gray], [0], None, [num_bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    histograms.append(("Gray", num_bins, hist))

# 初始化一个Matplotlib Figure


# # 绘制RGB和HSV直方图
# for color_space, num_bins, hist in histograms:
#     if color_space == "Gray":
#         continue
#
#     if color_space == "RGB":
#         plt.subplot(2, 3, 1)
#         plt.title("RGB Histograms")
#         plt.bar(np.arange(num_bins), hist[:num_bins], alpha=0.5, label="Channel 0")
#         plt.bar(np.arange(num_bins), hist[num_bins:2*num_bins], alpha=0.5, label="Channel 1")
#         plt.bar(np.arange(num_bins), hist[2*num_bins:], alpha=0.5, label="Channel 2")
#         plt.xlabel("Bin")
#         plt.ylabel("Frequency")
#         plt.legend()

# 绘制灰度直方图
plt.figure(figsize=(12, 6))
g = 0
h = 0
r = 0
for i, (color_space, num_bins, hist) in enumerate(histograms):
    if color_space == "Gray":
        g += 1
        plt.subplot(3, 3, g)
        plt.title("Gray Histogram {}".format(g))
        plt.hist(hist)
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
    elif color_space == "RGB":
        r += 1
        plt.subplot(3, 3, r+3)
        plt.title("RGB Histogram {}".format(r))
        plt.hist(hist)
        print(hist)
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        # plt.legend()
    elif color_space == "HSV":
        h += 1
        plt.subplot(3, 3, h+6)
        plt.title("HSV Histogram {}".format(h))
        plt.hist(hist)
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        # plt.legend()

plt.tight_layout()
plt.show()