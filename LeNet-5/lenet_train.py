#导入相应模块
import torch
import torch.nn as nn
from lenet_net import myletnet5
from torch.optim import lr_scheduler  #学习率应用
from torchvision import datasets,transforms #导入数据集
import os

#数据转换为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])
#加载训练的数据集----这里是调用的pytorch的数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = 32,shuffle = True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = 32,shuffle = True)
#加载测试集数据集


#如果有显卡，可以使用gpu
device = "cuda"if torch.cuda.is_available() else "cpu"
#调用之前搭建好的网络模型，将模型数据放在gpu
model = myletnet5().to(device)
#定义损失函数---交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
#定义一个优化器
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
#学习率每隔十轮变为原来的0.1倍
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)


#定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n = 0.0,0.0,0
    #前向传播
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        output = model(X)
        cur_loss = loss_fn(output,y)
        _,pred = torch.max(output,axis = 1)
        cur_acc = torch.sum(y==pred)/output.shape[0]
        # print(output.shape[0])
        print(cur_loss)
        # print(cur_acc)
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # print(batch)
        loss +=cur_loss.item()
        current +=cur_acc.item()
        n+=1
    print("train_loss"+str(loss/n))
    print("train_acc"+str(current/n))











#验证函数
def vail(dataloader,model,loss_fn):
    model.eval()
    loss ,current,n = 0.0,0.0,0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]


            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1
        print("val_loss" + str(loss / n))
        print("val_acc" + str(current / n))
        return current/n




#开始训练
epoch = 50
max_cur = 0
for t in range(epoch):
    print("-------------第"+str(t)+"轮----------------")
    train(train_dataloader,model,loss_fn,optimizer)
    a = vail(test_dataloader,model,loss_fn)
    if a>max_cur:
        path = "save_model"
        if not os.path.exists(path):
            os.mkdir(path)
        max_cur = a
        print("save best model!")
        torch.save(model.state_dict(),"save_model/best_model.pth")
print("done")
