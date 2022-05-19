#导入os库
import os
#导入Dataset库
from torch.utils.data import Dataset
#导入transforms
from torchvision import transforms
#设置transforms---将变量转换为张量
from utils import *
tf = transforms.Compose([
    transforms.ToTensor()
])
#构建一个类
class Mydata(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'VOC2012/SegmentationClass'))
#初始化----设定data的地址，获取原图所在文件夹的地址，获取该文件夹内图片的listdir----这里获取的是seg..clas
    def __len__(self):
        return len(self.name)
#定义长度获取
    def __getitem__(self, index):
        img_name = self.name[index]
        seg_path = os.path.join(self.path,'VOC2012/SegmentationClass',img_name)
        img_path = os.path.join(self.path,'VOC2012/JPEGImages',img_name.replace('png','jpg'))
        img = make_img_size(img_path)
        seg_img = make_img_size(seg_path)

        return tf(img),tf(seg_img)

#定义项目获取函数
#获取对应下标的数据
#对标签进行拼接地址
#拼接地址--获取原图的地址-----image----原图的地址
#图片大小不同----新建一个文件，构造函数，完成图片大小的填充
#这里不仅要把原图，还要把标签的图片进行同时填充

#返回transforms之后的两张图片

if __name__ =='__main__':
    data = Mydata('/home/rpf/nn_learn/u_net/data')
    print(data[1][0].shape)
    print(data[1][1].shape)
    from torch.nn.functional import one_hot
    # out = one_hot(data[0][1].long())
    # print(out.shape)

