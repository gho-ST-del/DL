import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture(r'E:\PythonCode\DeepLearning\CV\data\test.mp4')

# 读取第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建随机颜色
color = np.random.randint(0, 255, (100, 3))

# 定义Shi-Tomasi角点检测参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

# 定义Lucas-Kanade光流法参数
lk_params = dict(winSize=(60, 60),
                 maxLevel=6,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 检测第一帧的角点
p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)

# 创建掩膜用于绘制光流
mask = np.zeros_like(frame1)

while True:
    # 读取当前帧
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

    # 选择好的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制光流
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame2, mask)

    # 显示结果
    cv2.imshow('frame', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # 更新前一帧图像和角点
    prvs = next.copy()
    p0 = good_new.reshape(-1, 1, 2)

# 释放资源
cap.release()
cv2.destroyAllWindows()