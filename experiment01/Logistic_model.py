
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_size = 50
n_data= torch.ones(data_size, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
y1 = torch.zeros(data_size)  # 类型0 shape=(50, 1)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(data_size)  # 类型1 shape=(50, 1)
# 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat是合并数据)
x = torch.cat((x1, x2), 0).type(torch.FloatTensor).to(device)
y = torch.cat((y1, y2), 0).type(torch.FloatTensor).to(device)
test_n_data= torch.ones(data_size, 2)
test_x = torch.cat(( torch.normal(2 * test_n_data, 1),  torch.normal(-2 * test_n_data, 1)), 0).type(torch.FloatTensor).to(device)
test_y = torch.cat((torch.zeros(data_size), torch.ones(data_size)), 0).type(torch.FloatTensor).to(device)
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)]).to(device)
        yield features.index_select(0,j),labels.index_select(0,j)
# 训练
class Net(nn.Module):
    def __init__(self,input,output):
        super(Net, self).__init__()
        self.linear = nn.Linear(input, output)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
lr = 0.01
num_epochs = 300
batch_size = 10
model = Net(2,1).to(device)
loss =  nn.BCELoss()
optim = torch.optim.Adam(model.parameters(),lr= lr)
plt.figure(figsize=(10, 5))
losses = []
accuracies = []
for epoch in range(num_epochs):
    for X,Y in data_iter(batch_size,x,y):
        
        optim.zero_grad()
        y_hat =  model(X)
        los = loss(y_hat,Y.view(-1,1))
   
        los.backward()

        optim.step()
    res = model(test_x) 
    train_l = loss( res, test_y.view(-1,1))
    ress = res.view(-1).tolist()
    ys = y.tolist()
    acc = 0
    for i in range(len(ress)):
        t = 0
        if ress[i] > 0.3:
            t = 1
        if t == ys[i]:
            acc += 1
    accuracies.append(acc / len(y))
    losses.append(train_l.mean().item())
    plt.ion()
    # 动态显示损失值和准确率
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.pause(0.1)
    plt.ioff()
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x  