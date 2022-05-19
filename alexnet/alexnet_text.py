import torch
from alexnet import alexnet
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

#地址输入
TEST_PATH = "/home/rpf/神经网络学习/alexnet/data/test"


#将数据转化为tensor格式
transform = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
#定义数据的转换，将数据转换为张量，resize一下，归一化一下
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transform
])

#
test_dataset = ImageFolder(TEST_PATH,transform = test_transform)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True)

#使用gpu
device = 'cuda'if torch.cuda.is_available() else 'cpu'
#调用网络模型
model = alexnet().to(device)

#加载模型pt文件
model.load_state_dict(torch.load(('/home/rpf/神经网络学习/alexnet/save_model/best_model.pth')))

classes = [
    '猫',
    '狗'
]
show = ToPILImage()

#进入验证
model.eval()
for i in range(10):
    X,y = test_dataset[i][0],test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X,dim=0).float(),requires_grad = False).to(device)
    with torch.no_grad():
        pred = model(X)
        pred,actual = classes[torch.argmax(pred[0])],classes[y]
        print(f'预测："{pred}",目标："{actual}"')
