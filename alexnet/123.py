import torch
from torchvision import transforms
import cv2
num = cv2.imread(r'E:\神经网络学习\alexnet\data_name\Cat\0.jpg',0)
# print(num)
print(num.shape)
# cv2.imshow(num)
# cv2.waitKey(0)
# cv2.destroyWindow()
transform = transforms.ToTensor()
transform(num)
print(num)
