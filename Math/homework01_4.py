import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义函数 f(x, y) 和其梯度函数
def f(x, y):
    return (x*y - 1)**2

def gradient_f(x, y):
    return np.array([2*(x*y - 1)*y, 2*(x*y - 1)*x])
def gradient_descent(gradient_fn, learning_rate, num_iterations, initial_point):
    x = initial_point[0]
    y = initial_point[1]
    func_values = []
    gradient_trajectory = []
    for i in range(num_iterations):
        grad = gradient_fn(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        func_values.append(f(x, y))
        gradient_trajectory.append([x, y])

    return func_values, gradient_trajectory

learning_rate = 0.01
num_iterations = 20
initial_point = [3.0, 5.0]

func_values_0, gradient_trajectory_0 = gradient_descent(gradient_f, 0.01, num_iterations, initial_point)
func_values_1, gradient_trajectory_1 = gradient_descent(gradient_f, 0.02, num_iterations, initial_point)
func_values_2, gradient_trajectory_2 = gradient_descent(gradient_f, 0.04, num_iterations, initial_point)



# 绘制函数值随迭代过程的变化图
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), func_values_0, marker='o',color = 'red' ,label="lr=0.01")
plt.plot(range(num_iterations), func_values_1, marker='o',color = 'blue',label="lr=0.02")
plt.plot(range(num_iterations), func_values_2, marker='o',color = 'g'   ,label="lr=0.04")
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iterations')
plt.grid()
plt.legend()
plt.show()

# 绘制函数梯度的轨迹图在原函数表面上的表示
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

gradient_trajectory_0 = np.array(gradient_trajectory_0)
z_vals_0 = f(gradient_trajectory_0[:, 0], gradient_trajectory_0[:, 1])

gradient_trajectory_1 = np.array(gradient_trajectory_1)
z_vals_1 = f(gradient_trajectory_1[:, 0], gradient_trajectory_1[:, 1])

gradient_trajectory_2 = np.array(gradient_trajectory_2)
z_vals_2 = f(gradient_trajectory_2[:, 0], gradient_trajectory_2[:, 1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
ax.plot(gradient_trajectory_0[:, 0], gradient_trajectory_0[:, 1], z_vals_0, marker='o', color='red',label="lr=0.01")
ax.plot(gradient_trajectory_1[:, 0], gradient_trajectory_1[:, 1], z_vals_1, marker='x', color='b'  ,label="lr=0.02")
ax.plot(gradient_trajectory_2[:, 0], gradient_trajectory_2[:, 1], z_vals_2, marker='d', color='g'  ,label="lr=0.04")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Function Value')
ax.set_title('Gradient Trajectory on Function Surface')
plt.legend()
plt.show()
