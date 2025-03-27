import os
import numpy as np
import nibabel as nib



inputPathWT = 'postProcessing/outputWT'
inputPath3L = 'postProcessing/output3L'
outputPath = 'postProcessing/relabeled'

filesWT = [f.name for f in os.scandir(inputPathWT) if f.name.endswith('.nii.gz')]
files3L = [f.name for f in os.scandir(inputPath3L) if f.name.endswith('.nii.gz')]

for fileWT in filesWT:
    for file3L in files3L:
        if file3L == fileWT:
            filepathWT = os.path.join(inputPathWT,fileWT)
            filepath3L = os.path.join(inputPath3L,file3L)
            niftyWT = nib.load(filepathWT)
            nifty3L = nib.load(filepath3L)
            dataWT = niftyWT.get_fdata()
            data3L = nifty3L.get_fdata()

            new = np.zeros(dataWT.shape)

            #Redefine CC as NCR
            indexWT = dataWT == 1
            new[indexWT] = 2
            indexET = data3L == 1
            new[indexET] = 1
            indexCC = data3L == 2
            new[indexCC] = 3
            indexED = data3L == 3
            new[indexED] = 4
            saveNift = nib.Nifti1Image(new,niftyWT.affine)
            nib.save(saveNift,os.path.join(outputPath,file3L))