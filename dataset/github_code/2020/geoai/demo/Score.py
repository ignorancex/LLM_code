import cv2
import numpy as np

import argparse
import os

import cv2
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="input image path")
    #pass in the image folder
    parser.add_argument("-i", "--input", required=True,
                        help="input the image floder you want to score")
    #pass in the output file name (eg:result.txt)
    parser.add_argument("-o", "--output", required=True,
                        help="the name of output score file")

    return parser.parse_args()



#read image folder
def read_image(folder):
       for file in os.listdir(folder):
            path = os.path.join(folder, file)

#get score from user input
def getInt():
    temp=int(input("please enter an int 0-9"))
    if temp<0:
        getInt()
    elif temp>9:
        getInt()
    else:
        return temp



def main():
    args = parse_arguments()

    f=open(args.output,"a+")
    data=[]
    result=[]
    check=True
    kill=False
    temp=0
    for file in os.listdir(args.input):
         path = os.path.join(args.input, file)
         data.append(path)

#read each image and each image score input the output file
    for i in range(len(data)):
        image=cv2.imread(data[i])
        if image is not None:
            cv2.imshow(data[i],image)

            while(check):
                #set keyboard input
                k=cv2.waitKey()
                k=k-48
                #if user press 0-9
                if (0<=k<=9):
                    check=False
                    print(k)
                    mid = str(data[i]) + "\t" + str(k) + "\n"
                    result.append(mid)
                    cv2.destroyAllWindows()
                #if user press space
                elif (k==-16):
                    check=False
                    print("skiped picture")
                    cv2.destroyAllWindows()
                #user type esc will quit program
                elif(k==-21):
                    check=False
                    print("program finish")
                    cv2.destroyAllWindows()
                    kill=True
                #else loop back again
                else:
                    print("invaild input, try again")
                    check=True
            check=True
            if(kill==True):
                break


    for i in range(len(result)):
        f.write(result[i])

    f.close()



if __name__ == "__main__":
    main()
