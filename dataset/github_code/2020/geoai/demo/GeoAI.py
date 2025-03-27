import argparse
import sys
import os
import numpy as np
import cv2
from sklearn.externals import joblib

from helpers import clarity_rating


def main():
    """The main execution block of the program."""


    # Get user-provided information by parsing the program's arguments
    args = parse_arguments()

    # Load the SVM model from storage
    classifier = joblib.load(args.model)

    #store the folder
    folder=args.input
    range=int(args.range)

    #for loop go through the image folder
    for file in os.listdir(folder):

        path=os.path.join(folder, file)
        image=cv2.imread(path)
        #when image is not void start croping image to 200x200
        if image is not None:
            height=np.size(image,0)
            width=np.size(image,1)
            #if image size is less than 200x200, resize them to 200x200
            if height<200:
                image=cv2.resize(image,(200,200))
            if width<200:
                image=cv2.resize(image,(200,200))
            count=0.0
            total=0.0
            temp=0.0

            y=200
            x=200
            #each croped image will store as temo.jpg and get a score, this score will be store in the total
            while( y<height):
                while(x<width):
                    cropped=image[y-200:y, x-200:x]

                    if cropped is not None:
                        cv2.imwrite("temp.jpg",cropped)
                        hi=cv2.imread("temp.jpg")
                        if hi is not None:
                            prediction = classifier.predict(clarity_rating(hi))
                            count=count+1
                            if prediction=='0':
                                temp=0.0
                            if prediction=='1':
                                temp=1.0
                            if prediction=='2':
                                temp=2.0
                            if prediction=='3':
                                temp=3.0
                            if prediction=='4':
                                temp=4.0
                            if prediction=='5':
                                temp=5.0
                            if prediction=='6':
                                temp=6.0
                            if prediction=='7':
                                temp=7.0
                            if prediction=='8':
                                temp=8.0
                            if prediction=='9':
                                temp=9.0

                            total=total+temp
                            temp=0.0
                    x=x+200
                y=y+200
                x=0
            #when the average score less than score given by user, will be store in the good folder
            if total/count<range:
                temp="D:\good"
                cv2.imwrite(os.path.join(temp,file),image)
            #when the average score greater than score given by user, will be store in the Blur folder
            else:
                temp="D:\Blur"
                cv2.imwrite(os.path.join(temp,file),image)



def parse_arguments():
    """
    Parse the arguments provided to the program and make sure all information is present.
    :return: the arguments specified by the user
    """
    parser = argparse.ArgumentParser(description="Predict whether an image is blurry or not")

    parser.add_argument("-f", "--input", required=True,
                        help="The path of the image floder to classify")
    parser.add_argument("-r", "--range", required=True,
                        help="The range")
    parser.add_argument("-m", "--model", required=True,
                        help="The path of the SVM model file")

    return parser.parse_args()


# Start the program's execution at main() if run directly
if __name__ == "__main__":
    main()
