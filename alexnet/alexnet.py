import torch
import torch.nn as nn
import numpy as np

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2)
        self.p3 = nn.MaxPool2d(kernel_size=2)
        self.c4 = nn.Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.p5 = nn.MaxPool2d(2)
        self.c6 = nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.c7 = nn.Conv2d(in_channels=192,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.p8 = nn.MaxPool2d(kernel_size=4,padding=2)
        self.f = nn.Flatten()
        self.l9 = nn.Linear(2048,2048)
        self.l10 = nn.Linear(2048,1000)
        self.l11 = nn.Linear(1000,2)

    def forward(self,x):
        # print(x.shape)
        x = self.relu(self.c1(x))
        # print(x.shape)
        x = self.c2(x)
        # print(x.shape)
        x = self.p3(x)
        # print(x.shape)
        x = self.c4(x)
        # print(x.shape)
        x = self.p5(x)
        # print(x.shape)
        x = self.c6(x)
        # print(x.shape)
        x = self.c7(x)
        # print(x.shape)
        x = self.p8(x)
        # print(x.shape)
        x = self.f(x)
        # print(x.shape)
        x = self.l9(x)
        # print(x.shape)
        x = self.l10(x)
        # print(x.shape)
        output = self.l11(x)
        return output


# if __name__=="__main__":
#     x = torch.rand(1,3,224,224)
#     # x = np.random.rand(1,3,224,224)
#     model = alexnet()
#     out = model(x)
#     print(out)
#     print(out.shape)
#     print(out.shape[0])
