import skimage
from PIL import Image, ImageSequence, ImageEnhance
import cv2
import TraditionalAlgorithm
from data import *
import integration

def preprocessing(imageAddr,saveAddre=""):
    img = Image.open(imageAddr)
    images = []
    # read tif by slice
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page.load()
        page = page.convert('RGB')
        page = color_enhence(page)
        images.append(page)
    blendImages = []
    for i in range(len(images) - 3):
        im12 = Image.blend(images[i], images[i + 1], 0.5)
        im34 = Image.blend(images[i + 2], images[i + 3], 0.5)
        im = Image.blend(im12, im34, 0.5)
        blendImages.append(color_enhence(im,1.5))
    splitImages = []
    for i in range(6,len(blendImages)-7):
        for j in range(48):
            x = ((j%14)+1)*256
            y = (j%4)*256
            box = (x,y,x+256,y+256)
            split = blendImages[i].crop(box)
            splitImages.append(split)
            print(split.size)

    # save images to PNG (still have comments)
    for i in range(len(splitImages)):
        addr = saveAddre + str(i) + ".png"
        splitImages[i].save(addr)

def color_enhence(image,mul_index=2):
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(mul_index)
    return image_brightened

def col_enhence_Addr(imageAddr,saveAddre="",mul_index=2):
    img = Image.open(imageAddr)
    images = []
    # read tif by slice
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page.load()
        page = page.convert('L')
        enh_bri = ImageEnhance.Brightness(page)
        image_brightened1 = enh_bri.enhance(mul_index)
        images.append(page)

def resize(Addresses,saveAddre="",target_size=255):

    images = []
    for imageAddr,start,end in Addresses:
        img = Image.open(imageAddr)
        image = skimage.img_as_float(img)
        max = np.max(image)
        max = max if max>0 else 1
        for i, page in enumerate(ImageSequence.Iterator(img)):
            if (i >= start - 1 and i <= end - 1):
                page.load()
                page = page.convert('L')
                width = page.size[0] / target_size
                height = page.size[1] / target_size
                resize_index = width if width > height else height
                if (resize_index>1):
                    width = int(page.size[0] / resize_index)
                    height = int(page.size[1] / resize_index)
                    page = page.resize((width, height), Image.ANTIALIAS)
                else:
                    width = page.size[0]
                    height = page.size[1]
                valid_page = Image.new('L', (target_size, target_size))
                col = int((target_size - width) / 2)
                row = int((target_size - height) / 2)
                valid_page.paste(page, (col, row))
                images.append(valid_page)

    # save images to PNG (still have comments)
    for i in range(len(images)):
        addr = saveAddre + str(i) + ".png"
        images[i].save(addr)


