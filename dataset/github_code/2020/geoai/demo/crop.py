import argparse
import os

import cv2
import numpy as np

"""this crop program is use for crop all image in given folder and store them as 0.jpg,1.jpg...  """
def parse_arguments():
    parser = argparse.ArgumentParser(description="input an image file")

    parser.add_argument("-i", "--go", required=True,
                        help="the image files")

    return parser.parse_args()



#read image folder and crop each image to muti 200x200 picture
def read_image(floder):
    data=[]
    count=0
    for file in os.listdir(floder):
        path = os.path.join(floder, file)
        data.append(path)


    for i in range(len(data)):
        image =cv2.imread(data[i])
        height=np.size(image,0)
        width=np.size(image,1)

        y=300
        x=200
        while( y<height):
            while(x<width):
                cropped=image[y-200:y, x-200:x]
                temp=str(count)+".jpg"
                count=count+1
                if cropped[199] is not None:
                    cv2.imwrite(temp,cropped)
                x=x+200
            y=y+200
            x=0






def main():
    args = parse_arguments()

    read_image(args.go)

if __name__ == "__main__":
    main()
