from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from U_Net import *
import os
from torchvision.utils import save_image

device = 'cuda'if torch.cuda.is_available() else 'cpu'

weight_path = 'models/u_net.pth'

data_path = '/home/rpf/nn_learn/u_net/data'

save_path = '/home/rpf/nn_learn/u_net/data/train_image'

if __name__ =='__main__':
    data_loader = DataLoader(Mydata(data_path),batch_size=1,shuffle=True)
    net = U_Net(3).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load model')
    else:
        print("can not load model")
    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()


    epoch = 1
    while 1:
        for i ,(image,seg_img) in enumerate(data_loader):
            image,seg_img = image.to(device),seg_img.to(device)
            print(image.shape)
            out_img = net(image)
            train_loss = loss_fun(out_img,seg_img)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 5 times print loss
            if i%5 ==0:
                print(f'{epoch}-{i}-train_loss=====>>{train_loss.item()}')
            # 50 times save model
            if i%50 ==0:
                torch.save(net.state_dict(),weight_path)
                print('successfully save model')

            _image = image[0]
            # print(_image.shape)
            _seg_img = seg_img[0]*255
            # print(_seg_img.shape)
            # print(_seg_img)
            # _out_img = torch.argmax(out_img[0],dim=0).unsqueeze(0)*255
            _out_img = out_img[0]*255
            # print(_out_img.shape)
            # print(out_img)
            img = torch.stack([_image,_seg_img,_out_img],dim=0)
            save_image(img,f'{save_path}/{i}.jpg')
        epoch +=1