import numpy as np
import cv2
import matplotlib.pyplot as plt
# Load Frames
fr1 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12_balanced.png')
fr2 = cv2.imread(r'E:\PythonCode\DeepLearning\CV\data\12.png')

cv2.imshow('Frame 1', fr1)
cv2.imshow('Frame 2', fr2)
cv2.waitKey(0)

im1t = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY).astype(np.float32)
im2t = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY).astype(np.float32)

im1 = cv2.resize(im1t, None, fx=0.5, fy=0.5)
im2 = cv2.resize(im2t, None, fx=0.5, fy=0.5)

# Implementing Lucas Kanade Method
ww = 45
w = int(ww/2)

# Lucas Kanade Here
# for each point, calculate I_x, I_y, I_t
Ix_m = cv2.filter2D(im1, -1, np.array([[-1, 1], [-1, 1]]))  # partial on x
Iy_m = cv2.filter2D(im1, -1, np.array([[-1, -1], [1, 1]]))  # partial on y
It_m = cv2.filter2D(im1, -1, np.ones((2, 2))) + cv2.filter2D(im2, -1, -np.ones((2, 2)))  # partial on t
u = np.zeros_like(im1)
v = np.zeros_like(im2)

# within window ww * ww
for i in range(w, Ix_m.shape[0]-w):
    for j in range(w, Ix_m.shape[1]-w):
        Ix = Ix_m[i-w:i+w+1, j-w:j+w+1].flatten()
        Iy = Iy_m[i-w:i+w+1, j-w:j+w+1].flatten()
        It = It_m[i-w:i+w+1, j-w:j+w+1].flatten()

        A = np.vstack((Ix, Iy)).T
        b = -It.reshape((-1, 1))

        nu = np.linalg.pinv(A) @ b
        u[i, j] = nu[0,0]
        v[i, j] = nu[1,0]

# downsize u and v
u_deci = u[::10, ::10]
v_deci = v[::10, ::10]

# get coordinate for u and v in the original frame
m, n = im1t.shape
X, Y = np.meshgrid(np.arange(n), np.arange(m))
X_deci, Y_deci = X[::20, ::20], Y[::20, ::20]

# Plot optical flow field
plt.imshow(cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB))
plt.quiver(X_deci, Y_deci, u_deci, v_deci, color='y')
plt.show()