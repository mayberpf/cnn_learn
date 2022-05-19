from PIL import Image


def make_416_image(image_path):
    img = Image.open(image_path)
    w,h = img.size[0],img.size[1]
    # print(w,h)
    temp = max(w,h)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    return mask



if __name__ =='__main__':
    mask = make_416_image('data/images/000001.jpg')
    mask.show()