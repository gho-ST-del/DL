import torch


def init_1():
    M = torch.eye(1, 3)
    N = torch.ones(2, 1)
    return M, N


def sub_0(M, N):
    return M - N


def sub_1(M, N):
    return torch.sub(M, N)


def sub_2(M, N):
    return M.sub(N)


def one():
    M, N = init_1()
    print(M)
    print(N)
    print('第一种方法结果为:\n', sub_0(M, N))
    print('第二种方法结果为:\n', sub_1(M, N))
    print('第三种方法结果为:\n', sub_2(M, N))


def two():
    P = torch.rand(3, 2)
    P = P * 0.01 + 0
    Q = torch.rand(4, 2)
    Q = Q * 0.01 + 0
    Q_t = Q.t()
    print("PQ^t:\n", torch.mm(P, Q_t))


def break_off_1():
    x = torch.tensor(1.0, requires_grad=True)
    y1 = x ** 2
    # 上下文管理器
    with torch.no_grad():
        y2 = x ** 3
    y3 = y1 + y2
    y3.backward()
    print("中断方法1：", x.grad)


def break_off_2():
    x = torch.tensor(1.0, requires_grad=True)
    y1 = x ** 2
    y2 = (x ** 3).detach()
    y3 = y1 + y2
    y3.backward()
    print("中断方法2：", x.grad)


def three():
    break_off_1()
    break_off_2()


if __name__ == '__main__':
    # print("第一个小实验\n______")
    # one()
    # print("______")
    # print("第2个小实验\n______")
    # two()
    # print("______")
    print("第3个小实验\n______")
    three()
    print("______")
