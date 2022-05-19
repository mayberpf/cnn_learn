import torch
import torch.nn as nn
from yolov3_net import *
import os
from PIL import Image , ImageDraw
from utils import *
from torchvision import transforms
from config import *

class_num = {
    0:'person',
    1:'horse',
    2:'bicycle'
}

device = 'cpu'
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.net = YOLO_net().to(device)
        self.weights = '/home/rpf/nn_learn/yolov3/models/best.pth'
        self.net.load_state_dict(torch.load(self.weights))


        self.net.eval()
        # bi xu jia         sou yi xia   fang zhi guo ni he

    def forward(self,input,thresh,anchors,case):
        output_13,output_26,output_52 = self.net(input)

        index_13,bias_13 = self.get_index_and_bias(output_13,thresh)
        boxes_13 = self.get_true_position(index_13,bias_13,32,anchors[13],case)

        index_26, bias_26 = self.get_index_and_bias(output_26, thresh)
        boxes_26 = self.get_true_position(index_26,bias_26,16,anchors[26],case)

        index_52, bias_52 = self.get_index_and_bias(output_52, thresh)
        boxes_52 = self.get_true_position(index_52,bias_52,8,anchors[52],case)

        print('index_13--shape:'+str(index_13.shape))
        print('index_26--shape:'+str(index_26.shape))
        print('index_52--shape:'+str(index_52.shape))


        print('box_13--shape:'+str(boxes_13.shape))
        print('box_26--shape:'+str(boxes_26.shape))
        print('box_52--shape:'+str(boxes_52.shape))

        return torch.cat([boxes_13,boxes_26,boxes_52],dim=0)


    def get_index_and_bias(self,output,thresh):
        output = output.permute(0,2,3,1)
        output = output.reshape(output.size(0),output.size(1),output.size(2),3,-1)
        print('output--shape:'+str(output.shape))
        #n h w 3 (5+3)
        mask = torch.sigmoid(output[...,0]) > thresh
        print('mask--shape:'+str(mask.shape))
        index = mask.nonzero()
        print('index--shape:'+str(index.shape))
        bias = output[mask]
        print('index--shape:'+str(index.shape))
        print('bias--shape:'+str(bias.shape))
        return index,bias

    def get_true_position(self,index,bias,t,anchors,case):
        #case shi tu pian de suo fang bei shu
        anchors = torch.Tensor(anchors)

        a = index[:,3]

        cy = (index[:,1].float()+bias[:,2].float())*t/case
        cx = (index[:,2].float()+bias[:,1].float())*t/case


        w = anchors[a,0]*torch.exp(bias[:,3])/case
        h = anchors[a,1]*torch.exp(bias[:,4])/case

        p = bias[:,0]
        cls = bias[:,5:]

        cls_index = torch.argmax(cls,dim=1)

        return torch.stack([torch.sigmoid(p),cx,cy,w,h,cls_index],dim=1)



if __name__ =='__main__':
    detector = Detector()
    img = Image.open('/home/rpf/nn_learn/yolov3/data/images/001183.jpg')
    _img = make_416_image('/home/rpf/nn_learn/yolov3/data/images/001183.jpg')
    temp = _img.size[0]
    print(temp)
    _img = _img.resize((416,416))

    case = 416/temp
    print(case)
    tf = transforms.Compose([
        transforms.ToTensor()
        ])

    _img = tf(_img)
    _img = torch.unsqueeze(_img,dim=0)
    result=detector(_img,0.4,ANCHORS,case)
    draw = ImageDraw.Draw(img)

    print(result.shape)

    for ret in result:

        x1,x2,y1,y2 =ret[1]-0.5*ret[3],ret[1]+0.5*ret[4],ret[2]-0.5*ret[4],ret[2]+0.5*ret[4]
        print(x1,y1,x2,y2)

        print('class:',class_num[int(ret[5])])

        draw.text((x1,y1),str(class_num[int(ret[5].item())])+str(ret[0].item())[:4])

        draw.rectangle((x1,y1,x2,y2),outline='red',width=1)


    img.show()




