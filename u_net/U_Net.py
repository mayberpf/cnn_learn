#定义网络--导入--nn--
import torch
import torch.nn as nn
from torch.nn import functional as F
#构建layer类----两个卷积
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),#p  ---> kill node rate
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),  # p  ---> kill node rate
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class Down(nn.Module):
    def __init__(self,channel):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=2,stride=2,padding=0)
        )
    def forward(self,x):
        return self.layer(x)


class UP(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UP, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,x,feature):
        x = F.interpolate(x,scale_factor=2, mode='nearest')
        return torch.cat((self.layer(x),feature),dim=1)

class U_Net(nn.Module):
    def __init__(self,num_class):
        super(U_Net, self).__init__()

        self.c1 = Conv_Block(3,64)
        self.d2 = Down(64)
        self.c3 = Conv_Block(64,128)
        self.d4 = Down(128)
        self.c5 = Conv_Block(128,256)
        self.d6 = Down(256)
        self.c7 = Conv_Block(256,512)
        self.d8 = Down(512)
        self.c9 = Conv_Block(512,1024)
        self.up10 = UP(1024,512)
        self.c11 = Conv_Block(1024,512)
        self.up12 = UP(512,256)
        self.c13 = Conv_Block(512,256)
        self.up14 = UP(256,128)
        self.c15 = Conv_Block(256,128)
        self.up16 = UP(128,64)
        self.c17 = Conv_Block(128,64)
        self.out = nn.Conv2d(64,num_class,kernel_size=1,stride=1,padding=0)





    def forward(self,x):

        x_4 = self.c1(x)
        x_3 = self.c3(self.d2(x_4))
        x_2 = self.c5(self.d4(x_3))
        x_1 = self.c7(self.d6(x_2))
        x = self.c9(self.d8(x_1))
        x = self.up10(x,x_1)
        x = self.c11(x)
        x = self.up12(x,x_2)
        x = self.c13(x)
        x = self.up14(x,x_3)
        x = self.c15(x)
        x = self.up16(x,x_4)
        x = self.c17(x)
        out = self.out(x)


        return out
#初始化
#前向传播

#构建下采样----论文用的是池化，但是他采用的是步长为2的卷积

#上采样----转置卷积----差值法（这里看一下）

#在上采样之后，return的是torch.cat(两个特征图的拼接。在dim=1的维度上拼接)
#这里注意，这两个特征图的h，w一定是一样的

#定义整体网络
#初始化
#前向传播


#sigmod和softmax的区别（查一下）

if __name__ =='__main__':
    x = torch.randn(1,3,512,512)
    print(x.shape)
    net = U_Net(3)
    x = net(x)
    print(x.shape)
