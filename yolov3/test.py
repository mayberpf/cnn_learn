import torch


x = torch.randn(1,13,13,3,8)

mask = x[...,0]>1
# print(mask.shape)
# print(mask)
index = mask.nonzero()
print(index.shape)
# print(index)
print(x[mask].shape)
# print(x[mask])

/home/rpf/anaconda3/bin/python3.9 /home/rpf/神经网络学习/yolov3/test.py
torch.Size([67, 4])
torch.Size([67, 8])



/home/rpf/anaconda3/bin/python3.9 /home/rpf/神经网络学习/yolov3/detector.py
index--shape:torch.Size([0, 4])
bias--shape:torch.Size([0, 8])
index--shape:torch.Size([0, 4])
bias--shape:torch.Size([0, 8])
index--shape:torch.Size([0, 4])
bias--shape:torch.Size([0, 8])
index_13--shape:torch.Size([0, 4])
index_26--shape:torch.Size([0, 4])
index_52--shape:torch.Size([0, 4])
box_13--shape:torch.Size([0, 6])
box_26--shape:torch.Size([0, 6])
box_52--shape:torch.Size([0, 6])
torch.Size([0, 6])