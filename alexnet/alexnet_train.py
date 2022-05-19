#导入相应模块
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
import os
from alexnet import alexnet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
#这两行代码解决plt图片显示中中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#写训练测试的文件路径
TRAIN_PATH = 'E:/神经网络学习/alexnet/data/train'
VAL_PATH = 'E:/神经网络学习/alexnet/data/val'

#图片像素归一化[-1,1]之间
transform = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

#定义数据的转换，将数据转换为张量，resize一下，归一化等等
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transform
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transform
])

train_dataset = ImageFolder(TRAIN_PATH,train_transform)
test_dataset = ImageFolder(VAL_PATH,val_transform)
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=True)


batch_size = 16
#使用gpu的代码
device = 'cuda'if torch.cuda.is_available() else 'cpu'

#调用网络模型
model = alexnet().to(device)
#定义损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
#学习率下降
lr = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)


#定义训练函数
def train(d_loader,model,loss_fn,optimizer):

    loss,current,n = 0.0,0.0,0
    for batch,(X,y) in enumerate(d_loader):
        # for i in tqdm(range()):
        #     pass
        img,y = X.to(device),y.to(device)
        output = model(img)
        cur_loss = loss_fn(output,y)
        _,pred = torch.max(output,axis = 1)
        cur_acc = torch.sum(y==pred)/output.shape[0]
        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss+=cur_loss.item()
        current+=cur_acc.item()
        n+=1
    print("train_loss"+str(loss/n))
    print("train_acc"+str(current/n))
    return loss/n , current/n

#定义验证函数
def val(dataloader,model,loss_fn):
    model.eval()
    loss,current,n = 0.0,0.0,0
    with torch.no_grad():
        for idex,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            output = model(X)
            cur_loss = loss_fn(output,y)
            _,pred = torch.max(output,axis = 1)
            cur_acc = torch.sum(y==pred)/output.shape[0]

            loss+=cur_loss.item()
            current+=cur_acc.item()
            n+=1
    print("train_loss"+str(loss/n))
    print("train_acc"+str(current/n))
    return loss/n , current/n


#定义画图函数
def matplot_loss(train_loss,val_loss):
    plt.plot(train_loss,label = 'train_loss')
    plt.plot(val_loss,label = 'val_loss')
    plt.legend(loc = 'best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_curr(train_curr,val_curr):
    plt.plot(train_curr,label = 'train_curr')
    plt.plot(val_curr,label = 'val_curr')
    plt.legend(loc = 'best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集curr值对比图")
    plt.show()
#开始训练
epoch = 20
max_cur = 0.0
train_loss = []
train_curr = []
test_loss = []
test_curr = []
for i in range(epoch):
    print("-------------第" + str(i) + "轮----------------")
    t_loss,t_curr = train(train_loader,model, loss_fn, optimizer)
    v_loss,v_curr = val(test_loader,model, loss_fn)
    train_curr.append(t_curr)
    train_loss.append(t_loss)
    test_curr.append(v_curr)
    test_loss.append(v_loss)


    if v_curr > max_cur:
        path = "save_model"
        if not os.path.exists(path):
            os.mkdir(path)
        max_cur = v_curr
        print("save best model!")
        torch.save(model.state_dict(), "save_model/best_model.pth")
    if i ==epoch-1:
        torch.save(model.state_dict(), "save_model/last_model.pth")

print("done")





