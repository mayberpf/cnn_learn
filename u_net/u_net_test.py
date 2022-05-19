import os

from U_Net import *
import os
import torch
from utils import *
from data import *
from torchvision.utils import save_image


net = U_Net(3).cuda()
weight = '/home/rpf/nn_learn/u_net/models/model.pth'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
    print("successfully load model")
else:
    print('can not load model')

_input = input('please input the path of picture:')

image = make_img_size(_input)
image = tf(image).cuda()
image = torch.unsqueeze(image,dim=0)
out = net(image)
save_image(out,'/home/rpf/nn_learn/u_net/data/test_image/res.jpg')