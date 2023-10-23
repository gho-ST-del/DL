import cv2
import numpy as np

# 读取图像
img = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')

# 将图像转换为浮点数类型
img_float = img.astype(np.float32) 

# 计算图像的均值
img_mean = np.mean(img_float, axis=(0, 1))
K = (img_mean[0] + img_mean[1] + img_mean[2]) / 3.0
# 计算每个通道的增益系数
gain_b = K / img_mean[0]
gain_g = K / img_mean[1]
gain_r = K / img_mean[2]
print(gain_b)
print(gain_g)
print(gain_r)
# 对每个通道进行增益校正
img_balanced = img_float.copy()
img_balanced[:, :, 0] *= gain_b
img_balanced[:, :, 1] *= gain_g
img_balanced[:, :, 2] *= gain_r

# 将图像转换回整数类型
img_balanced = (img_balanced ).astype(np.uint8)
# print(img)
# print('-------------------')
# print(img_balanced)
# 将增益校正后的图像和初始图像合并在一起
img_concat = np.concatenate((img, img_balanced), axis=1)

# 显示合并后的图像
cv2.imshow('Original and Balanced', img_concat)
cv2.waitKey(0)
cv2.imwrite(r'E:\PythonCode\DeepLearning\CV\data\12_balanced.png', img_balanced)