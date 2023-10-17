import cv2
import numpy as np
img = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')

# print(img)
img_mean = cv2.mean(img)

b,g,r = cv2.split(img)


k = (img_mean[0]+img_mean[1]+img_mean[2])/3

gain_r = k/img_mean[2]
gain_g = k/img_mean[1]
gain_b = k/img_mean[0]

img[2] = gain_r*img[2]
img[1] = gain_g*img[1]
img[0] = gain_b*img[0]
# nr=np.clip(nr,0,255)
# ng=np.clip(ng,0,255)
# nb=np.clip(nb,0,255)

# new_img = cv2.merge((nb,ng,nr))

# print(new_img)
cv2.imshow('show',img)
cv2.waitKey(0)