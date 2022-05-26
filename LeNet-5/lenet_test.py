#导入相应模块
import torch
from lenet_net import myletnet5
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage
import os

#数据转换为tensor格式-----
data_transform = transforms.Compose([transforms.ToTensor()])
#加载训练的数据集----这里是调用的pytorch的数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = 16,shuffle = True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = 16,shuffle = True)
#加载测试集数据集

#如果有显卡，可以使用gpu ---使用cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
#调用之前搭建好的网络模型，将模型数据放在gpu
model = myletnet5().to(device)

#--------------做检测不需要这些---------------------#
#定义损失函数---交叉熵损失函数
# loss_fn = nn.CrossEntropyLoss()
#定义一个优化器
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
#学习率每隔十轮变为原来的0.1倍
# lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

#-------------加载模型文件：pth文件------------------#
model.load_state_dict(torch.load("/home/rpf/nn_learn/LeNet-5/save_model/best_model.pth"))

#获取结果
classes = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]
#张量转换为图片,方便可视化
show = ToPILImage()

#进入验证
for i in range(10):
    X,y = test_dataset[i][0],test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X,dim=0).float(),requires_grad = False).to(device)
    with torch.no_grad():
        pred = model(X)
        # print(pred[0])
        predicted , actual = classes[torch.argmax(pred[0])],classes[y]

        print(f'预测:"{predicted}",目标:"{actual}"')

