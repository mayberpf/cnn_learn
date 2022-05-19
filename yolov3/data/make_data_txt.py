import math
import xml.etree.ElementTree as ET
import os
from PIL import Image

class_num = {
    'person':0,
    'horse':1,
    'bicycle':2
}

xml_files = os.listdir('/home/rpf/nn_learn/yolov3/data/labels')#存放labels中所有的文件名
# print(xml_files)
with open('data.txt','a') as f:
    for xml_file in xml_files:
        # print(xml_file)
        tree =ET.parse(os.path.join('/home/rpf/nn_learn/yolov3/data/labels',xml_file))
        root = tree.getroot()
        image_name = root.find('filename')
        class_name = root.findall('object/name')
        boxes = root.findall('object/bndbox')
        # for x,y,w,h in boxes:
        #     print(x.text)
        # print(boxes)
        # for name,box in zip(class_name,boxes):
        #     cls = class_num[name.text]
        #     x1,y1,x2,y2 = box
        #     print(x1.text)
            # for x_min,y_min,x_max,y_max in box:

                # print(cls,x_min,y_min,x_max,y_max)

        #原图不是416*416的大小，所以需要进行缩放的计算
        file_name = image_name.text
        temp_size = max(Image.open(os.path.join('/home/rpf/nn_learn/yolov3/data/images',file_name)).size)
        # print(temp_size)
        ratio = 416/temp_size
        # print(ratio)
        data = []
        data.append(image_name.text)
        for cls,box in zip(class_name,boxes):
            cls = class_num[cls.text]
            cx,cy = math.floor((int(box[0].text)+int(box[2].text))/2),math.floor((int(box[1].text)+int(box[3].text))/2)
            w,h = math.floor(int(box[2].text)-int(box[0].text)),math.floor(int(box[3].text)-int(box[1].text))
            # print(cls,cx,cy,w,h )
            data.append(cls)
            data.append(cx*ratio)
            data.append(cy*ratio)
            data.append(w*ratio)
            data.append(h*ratio)
            # print(data)
        _str = ''
        for i in data:
            _str = _str + str(i) + ' '
        f.write(_str+'\n')
f.close()
