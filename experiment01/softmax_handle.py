import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1
num_epochs = 400
batch_size = 255
num_workers = 0
W = torch.tensor(np.random.normal(0, 0.01, (784, 10)), dtype=torch.float32).to(device)
b = torch.zeros(10, dtype=torch.float32).to(device)
W.requires_grad = True
b.requires_grad = True
eye = torch.eye(10)
class ProcessedFashionMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.mnist = torchvision.datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)
        self.targets = torch.eye(10)[self.mnist.targets]

    def __getitem__(self, index):
        image, target = self.mnist[index]
        processed_target = self.targets[index]
        return image, processed_target, target

    def __len__(self):
        return len(self.mnist)

mnist_train = ProcessedFashionMNIST(root='E:/毕设文件/DeepLearning/experiment01/Datasets/FashionMNIST', train=True,
                                    download=True, transform=transforms.ToTensor())
mnist_test = ProcessedFashionMNIST(root='E:/毕设文件/DeepLearning/experiment01/Datasets/FashionMNIST', train=False,
                                   download=True, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers)

print("数据集初始化完成")
def f_softmax(z):
    c = torch.max(z)
    exp_a = torch.exp(z - c)  # 溢出对策
    sum_exp_a = torch.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def softmax(X, w, b):
    x = torch.mm(X.view(X.size(0), -1),w) + b

    return f_softmax(x)
def cross_entropy(y_hat, y):
    log_softmax_out = torch.log(y_hat + 1e-7)  # 添加一个小的常数，避免对数中的零
    loss = -torch.sum(y * log_softmax_out) / len(y)
    return loss
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
net = softmax
loss = cross_entropy
print("开始训练")
losss=[]
accs=[]
plt.figure(figsize=(10, 5))
for epoch in range(num_epochs):
    for data in train_iter:
        imgs, targets, index = data
        y_hat = net(imgs.to(device), W, b)  # 模型的输出
        l = loss(y_hat, targets.to(device))  # 计算损失
        l.backward()
        sgd([W, b], lr, batch_size)
        W.grad.data.zero_()
        b.grad.data.zero_()
    losslist=[]
    acclist=[]
    for data in test_iter:
        imgs, targets, index = data
        y_hat = net(imgs.to(device), W, b)  # 模型的输出
        # print(y_hat)
        l = loss(y_hat, targets.to(device))  # 计算损失
        y_index = torch.argmax(y_hat, dim=1)
        acc = 0
        for i in range(len(y_index)):
            if y_index[i] == index[i]:
                acc += 1
        acclist.append(acc)
        losslist.append(l.item())
    loss_num = np.mean(losslist)*1
    acc_num = np.mean(acclist) / batch_size * 100
    losss.append(loss_num)
    accs.append(acc_num)
    print('epoch %d, loss %f acc: %f %%' % (epoch + 1,loss_num,acc_num))
    plt.ion()
    # 动态显示损失值和准确率
    plt.subplot(1, 2, 1)
    plt.plot(losss, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accs, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.pause(0.1)
    plt.ioff()
