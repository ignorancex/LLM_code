from torchvision.datasets.utils import download_file_from_google_drive

print('Test:')
download_file_from_google_drive('1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu', '.', 'VisDrone2019-MOT-val.zip')

print('\nTrain:')
download_file_from_google_drive('1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh', '.', 'VisDrone2019-MOT-train.zip')

print()