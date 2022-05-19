#导入相应模块
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.autograd import Variable
#定义一个模型
class myletnet5(nn.Module):
    # 建立类相应初始化
    def __init__(self):
        super(myletnet5,self).__init__()
        # 定义各个网络层
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sigmod = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.flatten = nn.Flatten()
        self.n6 = nn.Linear(120,84)
        self.n7 = nn.Linear(84,10)
    # 写一个前向传播函数
    def forward(self,x):
        x = self.c1(x)
        # print(x.shape)
        x = self.sigmod(x)
        # print(x.shape)
        x = self.s2(x)
        # print(x.shape)
        x = self.c3(x)
        # print(x.shape)
        x = self.s4(x)
        # print(x.shape)
        x = self.c5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.n6(x)
        # print(x.shape)
        x = self.n7(x)
        # print(x.shape)
        return x



#主函数，可以不写
#
# if __name__ =="__main__":
#     x = torch.rand(1,1,28,28)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     data_transform = transforms.Compose([transforms.ToTensor()])
#     test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
#     model = myletnet5()
#     x = test_dataset[0][0]
#     x = x.unsqueeze(0)
#     res = model(x)
#     print(res)
#     print(res.shape)
#     print(res.shape[0])