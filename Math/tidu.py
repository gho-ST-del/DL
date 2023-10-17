import torch
import numpy as np
import matplotlib.pyplot as plt


def func1(x,y):
    return  x**2 * y**2
def func2(x,y):
    return x**2+20*y**2+0.01*(x**2+y**2)**2
def func3(x,y):
    return x**2+0.01*(x**2+y**2)**2
def func4(x,y):
    return (x**2)*(y**2)
def func5(x,y):
    return (x*y-1)**2


def cal(lr,funcc):
    x = torch.ones(1, 1, dtype=torch.float)
    x *=3
    y = torch.ones(1, 1, dtype=torch.float)
    y *=4

    x.requires_grad_(requires_grad=True)
    y.requires_grad_(requires_grad=True)
    fold=1000000
    f=funcc
    fn=f(x,y)
    global t
    t=0
    lists = []
    ress=[]
    while(True):

        l=f(x,y)
        l.backward()

        x.data -= lr * x.grad
        y.data -= lr * y.grad
        x.grad.zero_()
        y.grad.zero_()
        fn=f(x,y)
        t+=1
        lists.append(l.item())

        ress.append([x.tolist()[0][0],y.tolist()[0][0]])
        print(t,x,y,fn)
        if(t>10):
            return lists,ress



#cal(x,y,1,func2)
#cal(x,y,1,func3)
#cal(x,y,1,func4)
#cal(x,y,1,func5)


def draw(list1,list2,list3,lu1,lu2,lu3,funn):
    plt.plot(list1, label="lr = 0.01")
    plt.plot(list2, label="lr = 0.02")
    plt.plot(list3, label="lr = 0.03")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = funn(X, Y)
    lu1 = np.array(lu1)

    z_vals_1 = funn(lu1[:, 0], lu1[:, 1])
    print(lu1[:, 0])
    lu2 = np.array(lu2)
    z_vals_2 = funn(lu2[:, 0], lu2[:, 1])
    lu3 = np.array(lu3)
    z_vals_3 = funn(lu3[:, 0], lu3[:, 1])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
    ax.plot(lu1[:, 0], lu1[:, 1], z_vals_1, marker='o', color='red',
            label="lr=0.01")
    ax.plot(lu2[:, 0], lu2[:, 1], z_vals_2, marker='x', color='b', label="lr=0.02")
    ax.plot(lu3[:, 0], lu3[:, 1], z_vals_3, marker='d', color='g', label="lr=0.04")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Function Value')
    ax.set_title('Gradient Trajectory on Function Surface')
    plt.legend()
    plt.show()

re1,lu1 = cal(0.001,func2)
re2,lu2 = cal(0.002,func2)
re3,lu3 = cal(0.003,func2)
draw(re1,re2,re3,lu1,lu2,lu3,func2)