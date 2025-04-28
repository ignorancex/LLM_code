import os
import pydicom as dicom
import numpy as np
import cv2
import argparse

def asciiart():
    """
    ASCII ART
    Reference: https://patorjk.com/software/taag/#p=testall&f=JS%20Stick%20Letters&t=ai-echo%0A
    """
    print(r"""

                                                           _..._                 .-'''-.     
                                                        .-'_..._''.             '   _    \   
             .--.                     __.....__       .' .'      '.\  .       /   /` '.   \  
             |__|                 .-''         '.    / .'           .'|      .   |     \  '  
             .--.                /     .-''"'-.  `. . '            <  |      |   '      |  ' 
        __   |  | ,.----------. /     /________\   \| |             | |      \    \     / /  
     .:--.'. |  |//            \|                  || |             | | .'''-.`.   ` ..' /   
    / |   \ ||  |\\            /\    .-------------'. '             | |/.'''. \  '-...-'`    
    `" __ | ||  | `'----------'  \    '-.____...---. \ '.          .|  /    | |              
     .'.''| ||__|                 `.             .'   '. `._____.-'/| |     | |              
    / /   | |_                      `''-...... -'       `-.______ / | |     | |              
    \ \._,\ '/                                                   `  | '.    | '.             
     `--'  `"                                                       '---'   '---'            


                    """)

def fast_scandir(dirname: str):
    """
    Function to scan a directory in a fast way
    """
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def mask(output: np.ndarray):
    """
    Mask pixels outside of scanning sector
    """
    dimension = output.shape[0]
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)
    return maskedImage

def ycrop_mean(testarray: np.ndarray):
    """
    yCrop mean
    TODO: how the yCrop mean is impacting on the file conversion?
    """
    frame0 = testarray[0]
    mean = np.mean(frame0, axis=1)
    mean = np.mean(mean)

    yCrop = np.where(mean<200)[0][0]
    testarray = testarray[:, yCrop:, :, :]

    bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
    if bias>0:
        if testarray.shape[1] < testarray.shape[2]:
            testarray = testarray[:, :, bias:-bias, :]
        else:
            testarray = testarray[:, bias:-bias, :, :]

    return testarray

def makeVideo(fileToProcess: str, destinationFolder: str, cropSize: list):
    try:
        fileName = fileToProcess.split('/')[-1]  # \\ if GNU/linux OS
        # fileName = fileToProcess.split('\\')[-1] #\\ if Windows OS
        # fileName = hex(abs(hash(fileToProcess.split('/')[-1]))).upper() ##/ if on mac or sherlock

        if not os.path.isdir(os.path.join(destinationFolder,fileName)):
            try:
                os.makedirs(destinationFolder)
            except FileExistsError:
                pass

            dataset = dicom.dcmread(fileToProcess, force=True)
            # print(dataset) # Print metadata
            print(f'    metaDICOM:  {dataset[0x0008, 0x1030]}')     # Study Description  LO: 'Lung / Cons/Eff'
            print(f'    metaDICOM:  {dataset[0x7fe0, 0x0010]}')     # Pixel Data
            print(f'    metaDICOM:  {dataset[0x0018, 0x0040]}')     # Cine Rate
            print(f'    metaDICOM:  {dataset[0x0018, 0x1242]}')     # Actual Frame Duration
            print(f'    metaDICOM:  {dataset[0x0018, 0x1063]}')     # Frame Time
            print(f'    metaDICOM:  {dataset[0x0018, 0x1066]}')     # Frame Delay
            print(f'    metaDICOM:  {dataset[0x0018, 0x1088]}')     # Heart Rate
            print(f'    metaDICOM:  {dataset[0x0028, 0x0008]}')     # Number of Frames
            print(f'    metaDICOM:  {dataset[0x0028, 0x0009]}')     # Frame Increment Pointer

            USframes_array = dataset.pixel_array
            frames,height,width,channels = USframes_array.shape
            print(f'  frames,height,width,channels = {USframes_array.shape}')

            fps = 30
            try:
                fps = dataset[(0x18, 0x40)].value
            except:
                print("couldn't find frame rate, default to 30")

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            video_filename = os.path.join(destinationFolder, fileName + '.avi')
            out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)

            for i in range(frames):
                outputA = USframes_array[i,:,:,0]
                smallOutput = outputA[0:int(height), 0:int(width)] #outputA[0:846, 0:1538]

                output = cv2.resize(smallOutput, cropSize, interpolation=cv2.INTER_CUBIC)
                # output = mask(output) # With mask
                out.write(cv2.merge([output, output,output]))

            out.release()

        else:
            print(fileName,"hasAlreadyBeenProcessed")
    except:
        print(":WARNING: Something failed, not sure what, have to debug", fileName)
    return 0


def main():
    asciiart()
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True, help='Specify the dataset path.')
    parser.add_argument('--participant_ID', required=False, help='Specify participant ID name.')
    parser.add_argument('--rawdatapath', required=False, help='Specify rawdataset path.')
    parser.add_argument('--preprocesseddatapath', required=False, help='Specify preprocesseddatapath path.')
    args = parser.parse_args()

    pathToProcess = os.path.join(args.datapath, "raw-datasets", args.participant_ID)
    destinationFolder = os.path.join(args.datapath, "preprocessed-datasets", args.participant_ID)

    number_of_paths=len(fast_scandir(pathToProcess))
    main_DICOMfiles_path=fast_scandir(pathToProcess)[int(number_of_paths/2):number_of_paths]

    cropSize = (1538, 846)
    for DICOMfiles_path_i in main_DICOMfiles_path:
        day_i=DICOMfiles_path_i[56:58]
        print(f'>>>>> Day:',{day_i})
        DICOMfiles_path_i_ = os.listdir(DICOMfiles_path_i)
        total_number_of_DICOMfiles = len(DICOMfiles_path_i_)

        count = 0
        for DICOMfile_i in DICOMfiles_path_i_:
            count += 1
            VideoPath_DICOMfile_i = os.path.join(DICOMfiles_path_i, DICOMfile_i)
            print(f' Day/DICOMfile_i:', {day_i, DICOMfile_i}, ';' , {count} , '/' , {total_number_of_DICOMfiles})
            destinationFolder_DICOMfile_i_ = os.path.join(destinationFolder, day_i)
            destinationFolder_DICOMfile_i_AVI_ = os.path.join(destinationFolder_DICOMfile_i_, DICOMfile_i + ".avi")
            if not os.path.exists(destinationFolder_DICOMfile_i_AVI_):
                makeVideo(VideoPath_DICOMfile_i, destinationFolder_DICOMfile_i_, cropSize)
            else:
                 print(f' :WARNING: Already converted', {DICOMfile_i}, 'to avi')

if __name__=='__main__':
    main()
