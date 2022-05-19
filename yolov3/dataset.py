import math
import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils import make_416_image
import numpy as np
from config import *



transform = transforms.Compose([
    transforms.ToTensor()
])



def one_hot(cls_num,i):
    res = np.zeros(cls_num)
    res[i] = 1.
    # print(res)
    return res


class YoloDataset(Dataset):
    def __init__(self):
        f = open('data/data.txt','r')
        self.dataset =f.readlines()

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data = self.dataset[index].split()
        boxes = np.array([float(x) for x in data[1:]])
        boxes = np.split(boxes,len(boxes)//5)
        # print(boxes)
        # print(data[0])
        image =make_416_image(os.path.join('data/images',data[0]))
        # print(image.size)
        img = image.resize((416,416))
        # print(img)
        img = transform(img)
        # print(img.shape)

        #image.size = 480*364------->image.size = 480*480------>resize (416*416)
        label = {}
        for feature_size ,anchors in ANCHORS.items():
            # print(feature_szie,anchors)
            label[feature_size] = np.zeros((feature_size,feature_size,3,5+CLASS_NUM))
            # print(label[feature_size].shape)
            #这里的13*13*3*8---->将图像提取到13*13个特征网络，每个网格预测三个anchor，每个anchor有3个类别概率和c,w,h,x,y
            for box in boxes:
                # print(box)
                cls,cx,cy,w,h = box
                # print(cls,cx,cy,w,h)
                _x,x_index = math.modf(cx*feature_size/DATA_WIDTH)
                _y,y_index = math.modf(cy*feature_size/DATA_WIDTH)
                for i , anchor in enumerate(anchors):
                    p_w , p_h = w/anchor[0],h/anchor[1]
                    #这里置信度的计算使用的是：小面积比大面积
                    area = w*h
                    iou = min(area,ANCHORS_GROUP_AREA[feature_size][i])/max(area,ANCHORS_GROUP_AREA[feature_size][i])

                    label[feature_size][int(x_index),int(y_index)] = np.array([iou,_x,_y,np.log(p_w),np.log(p_h),*one_hot(CLASS_NUM,int(cls))])

        return label[13],label[26],label[52],img

if __name__ =='__main__':
    dataset = YoloDataset()
    # # print(len(dataset))
    print(dataset[0][3].shape)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    # a = one_hot(3,1)
    # print(a)