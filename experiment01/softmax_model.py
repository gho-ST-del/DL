import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.01
num_epochs = 400
batch_size = 255
num_workers = 0
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
class MySoftmax(nn.Module):
    def __init__(self, input, output):
        super(MySoftmax, self).__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        out = nn.functional.softmax(x,dim=1)
        return out
net = MySoftmax(28*28,10).to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr = lr)
print("开始训练")
losss=[]
accs=[]
plt.figure(figsize=(10, 5))
for epoch in range(num_epochs):
    for data in train_iter:
        imgs, targets, index = data
        y_hat = net(imgs.to(device))  # 模型的输出
        l = loss(y_hat.view(-1), targets.view(-1).to(device))  # 计算损失
        opt.zero_grad()
        l.backward()
        opt.step()
    losslist = []
    acclist = []
    for data in test_iter:
        imgs, targets, index = data
        y_hat = net(imgs.to(device))# 模型的输出
        l = loss(y_hat.view(-1), targets.view(-1).to(device))  # 计算损失
        y_index = torch.argmax(y_hat, dim=1)
        acc = 0
        for i in range(len(y_index)):
            if y_index[i] == index[i]:
                acc += 1
        acclist.append(acc)
        losslist.append(l.item())
    loss_num = np.mean(losslist) * 1
    acc_num = np.mean(acclist) / batch_size * 100
    losss.append(loss_num)
    accs.append(acc_num)
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
    print('epoch %d, loss %f acc: %f %%' % (epoch + 1, np.mean(losslist), np.mean(acclist) / batch_size * 100))
