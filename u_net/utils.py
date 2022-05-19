#导入Image
from PIL import Image


def make_img_size(path):
    img = Image.open(path)
    max_side = max(img.size)
    # print(img.size)
    new = Image.new('RGB',(max_side,max_side),(0,0,0))
    new.paste(img,(0,0))
    new = new. resize((256,256))
    # new.show()
    return new

#zhe li jian yi jiang tu pian suo xiao ----->xiao guo hui hao
#这里构造一个函数---传入地址
#读取图片
#取得图片的最长边的大小
#构建一个由最长边为边长的正方形图片（黑色图片）
#将原图和新建的黑色图片进行拼接---其0，0点重合

#可以完成图片的形状转换为正方形




if __name__ =='__main__':
    img = make_img_size('/home/rpf/nn_learn/u_net/data/VOC2012/JPEGImages/2007_000032.jpg')
    print(img.size)
