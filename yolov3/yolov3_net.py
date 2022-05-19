import torch
import torch.nn as nn
from torch.nn import functional

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(ConvolutionalLayer,self).__init__()
        self.CBL = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.CBL(x)

class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()


    def forward(self,x):
        return functional.interpolate(x,scale_factor=2,mode='nearest')


class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self,x):
        return self.downsample(x)


class Res_unit(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Res_unit, self).__init__()
        self.res_unit = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1,1,0),
            ConvolutionalLayer(out_channels, in_channels, 3,1,1)
        )
    def forward(self,x):
        return self.res_unit(x)+x

class CBL_5(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(CBL_5, self).__init__()
        self.cbl_5 = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self,x):
        return self.cbl_5(x)

class Detect(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(Detect, self).__init__()
        self.detect = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernel_size, stride, padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
    def forward(self,x):
        return self.detect(x)

class Upsample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            UpSampleLayer()
        )
    def forward(self,x):
        return self.upsample(x)

class YOLO_net(nn.Module):
    def __init__(self):
        super(YOLO_net, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            Downsample(32, 64, 3, 2, 1),
            Res_unit(64, 32),
            Downsample(64, 128, 3, 2, 1),
            Res_unit(128, 64),
            Res_unit(128, 64),
            Downsample(128, 256, 3, 2, 1),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128),
            Res_unit(256, 128)
        )
        self.trunk_26 = nn.Sequential(
            Downsample(256, 512, 3, 2, 1),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256),
            Res_unit(512, 256)
        )
        self.trunk_13 = nn.Sequential(
            Downsample(512, 1024, 3, 2, 1),
            Res_unit(1024, 512),
            Res_unit(1024, 512),
            Res_unit(1024, 512),
            Res_unit(1024, 512)
        )
        self.convset_13 = nn.Sequential(
            CBL_5(1024,512)
        )
        self.detect_13 = nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,24,kernel_size=3,stride=1,padding=1)
        )
        self.up13_26 = nn.Sequential(
            Upsample(512,256)
        )
        self.convset_26 = nn.Sequential(
            CBL_5(768,256)
        )
        self.detect_26 = nn.Sequential(
            ConvolutionalLayer(256,512,3,1,1),
            nn.Conv2d(512,24,kernel_size=3,stride=1,padding=1)
        )
        self.up26_52 = nn.Sequential(
            Upsample(256,128)
        )
        self.convset_52 = nn.Sequential(
            CBL_5(384,128)
        )
        self.detect_52 = nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256,24,kernel_size=3,stride=1,padding=1)
        )



    def forward(self,x):
        out_52 = self.trunk_52(x)
        # print(out_52.shape)
        out_26 = self.trunk_26(out_52)
        # print(out_26.shape)
        out_13 = self.trunk_13(out_26)
        # print(out_13.shape)
        up_13 = self.convset_13(out_13)
        out_13 = self.detect_13(up_13)


        up13_26 = self.up13_26(up_13)
        out_26 = torch.cat((out_26,up13_26),dim = 1)
        up_26 = self.convset_26(out_26)
        out_26 = self.detect_26(up_26)

        up26_52 = self.up26_52(up_26)
        out_52 = torch.cat((out_52,up26_52),dim = 1)
        out_52 = self.convset_52(out_52)

        out_52 = self.detect_52(out_52)


        return out_13,out_26,out_52


if __name__ =="__main__":
    net = YOLO_net()
    x = torch.randn(1,3,416,416)
    out_13,out_26,out_52  = net(x)

    print(out_13.shape)
    print(out_26.shape)
    print(out_52.shape)
