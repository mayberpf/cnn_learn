import torch.cuda
from torch import optim
import torch.nn as nn
from  torch.utils.data import DataLoader
from dataset import *
from yolov3_net import *
import os

from torch.utils.tensorboard import SummaryWriter

def loss_fun(output,traget,c):
    output = output.permute(0,2,3,1)
    output = output.reshape(output.size(0),output.size(1),output.size(2),3,-1)
    #shu chu xing zhuang bian wei n*13*13*3*8

    mask_obj = traget[...,0]>0
    mask_no_obj = traget[...,0]==0

    loss_p_fun =  nn.BCELoss()
    loss_p = loss_p_fun(torch.sigmoid(output[...,0]),traget[...,0])

    loss_box_fun = nn.MSELoss()
    loss_box = loss_box_fun(output[mask_obj][...,1:5],traget[mask_obj][...,1:5])

    loss_class_fun = nn.CrossEntropyLoss()
    loss_class = loss_class_fun(output[mask_obj][...,5:],torch.argmax(traget[mask_obj][...,5:],dim=1,keepdim=True).squeeze(dim = 1))

    # loss_no_obj_fun = nn.BCELoss()
    # loss_no_obj = loss_no_obj_fun(output[mask_no_obj][...,0],traget[mask_no_obj][0])


    loss = c*loss_p+(1-c)*0.5*loss_box+(1-c)*0.5*loss_class

    return loss





if  __name__ =='__main__':

    summary_write = SummaryWriter('logs')

    device = 'cuda'if torch.cuda.is_available() else 'cpu'
    dataset = YoloDataset()
    data_loader = DataLoader(dataset,batch_size=2,shuffle=True)

    net = YOLO_net().to(device)
    weight_path = 'models/best.pth'
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))

    opt = optim.Adam(net.parameters())

    epochs = 200

    for epoch in range(epochs):

        for target_13,target_26,target_52,img_data in data_loader:
            target_13,target_26,target_52,img_data = target_13.to(device),target_26.to(device),target_52.to(device),img_data.to(device)
            output_13,output_26,output_52 = net(img_data)

            loss_13 = loss_fun(output_13.float(), target_13.float(),0.7)
            loss_26 = loss_fun(output_26.float(), target_26.float(), 0.7)
            loss_52 = loss_fun(output_52.float(), target_52.float(), 0.7)

            loss = loss_52+loss_26+loss_13
            print('loss = '+str(loss))
            summary_write.add_scalar('train_loss',loss,epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(net.state_dict(),'models/best.pth')
        print('save model success')

