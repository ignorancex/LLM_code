import argparse
import os

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

from helpers import clarity_rating


def main():
    """The main execution block of the program."""

    # Get user-provided information by parsing the program's arguments
    args = parse_arguments()

    # Create a default Support Vector Machine (SVM)
    classifier = SVC()

    #read score data
    with open(args.data,"r") as f:
        content = f.readlines()

    content = [x.strip() for x in content]


    i=0
    a=[]
    b=[]
    size=len(content)

    while(i<size):
        l,r=content[i].split("\t")
        a.append(l)
        b.append(r)
        i=i+1
    i=0
    f.close()


    # Read in all blurry and clear images
    data = read_images(a,b)

    # Create a list of clarity ratings by grabbing the first values from the tuples
    ratings = [x[0] for x in data]

    # Create a list of classifications by grabbing the second values from the tuples
    classifications = [x[1] for x in data]

    # Reshape the list of ratings so that the SVM doesn't complain
    reshaped_ratings = np.reshape(ratings, (-1,1))

    # Train the SVM on the clarity ratings and the blurriness classifications
    classifier.fit(reshaped_ratings, classifications)

    # Save the SVM model to storage
    joblib.dump(classifier, args.output)


def parse_arguments():
    """
    Parse the arguments provided to the program and make sure all information is present.
    :return: the arguments specified by the user
    """
    parser = argparse.ArgumentParser(description="Train a blur detection SVM with supplied images")

    #parser.add_argument("-f", "--floder", required=True,
                    #    help="The path of the folder containing blurr training images")
    parser.add_argument("-i", "--data", required=True,
                        help="The path of the file containing score")
    parser.add_argument("-o", "--output", required=True,
                        help="The path of the file to save the model to")


    return parser.parse_args()


def read_images(a,b):
    """
    Read in a folder of images and classify each image.
    :param folder: the path to the folder containing training images
    :param classification: the value to classify each image as
    :return: a list containing tuples with information about the images
    """




    # Temporary list for storing image information
    data = []

    size=len(a)


    i=0
    while(i<size):
        image=cv2.imread(a[i])
        data.append((clarity_rating(image), b[i]))
        i=i+1


    return data


# Start the program's execution at main() if run directly
if __name__ == "__main__":
    main()
