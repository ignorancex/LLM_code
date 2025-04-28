import os
import pandas as pd
from time import time
from timeit import default_timer as timer
from datetime import timedelta
from collections import defaultdict
from distutils.dir_util import copy_tree
from sklearn.model_selection import GroupKFold


def PatientToScanID():
    p_file = open("PatientDict.txt")
    d = defaultdict(list)
    # associates list of scanID (value) to key (patient number)
    for line in p_file:
        key, value = line.split()
        d[key].append(value)
    return d

# For NoReg resulting fields
def RigidAffineSynTransfoPair1_5():

    p_dict = PatientToScanID()

    times = []
    mov_fix_s = []

    start_big = timer()
    for p in p_dict:
        list_scanIDs = sorted(os.listdir(f"/media/andjela/SeagatePor/PairRegData/images/{p}/"))
        intra_sample_indices = []
        num_images_in_group = len(list_scanIDs)
        for i in range(num_images_in_group):
            for j in range(i):
                intra_sample_indices.append((j, i))
        num_pairs = len(intra_sample_indices)
        #Iteration over possible pairs for specific patient
        #access index of first image to register with intra_sample_indices[0][0] for first pair possible
        for i in range(num_pairs):
            moving = list_scanIDs[intra_sample_indices[i][0]] #gives specific scanID for moving image
            fixed = list_scanIDs[intra_sample_indices[i][1]] #gives specific scanID for fixed image
            if not os.path.exists(f'/media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/'): #[:-7] to remove .nii.gz from
                os.mkdir(f'/media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/')
            start = time.time()
            os.system(f"antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 " \
                f"--initial-moving-transform [ /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}, /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving}, 0 ] " \
                f"--initialize-transforms-per-stage 0 " \
                f"--interpolation Linear " \
                f"--output [ /media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_, /media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz, /media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/fix2mov_warped_image.nii.gz ] " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}, /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--transform Affine[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}, /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--transform SyN[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}, /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--write-composite-transform 1 " \
                f"--verbose 1 " 
                )
            end = time.time()

            # DDF creation from the h5 file
            os.system(f"antsApplyTransforms -d 3 -o [/media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/ddf_ANTs.nii.gz, 1] -v 1 -t /media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_Composite.h5 " \
                f"-r  /media/andjela/SeagatePor/NoReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}")
                
            mov_fix_s.append(f'{moving[:-7]}_{fixed[:-7]}')
            times.append(end-start)

    time_taken = {'scanIDs' : mov_fix_s ,  'times': times}
    df_timer = pd.DataFrame(data=time_taken)


    df_timer.to_csv(f"/media/andjela/SeagatePor/PairReg/rigid-affine-syn-1.5/images/time_taken")

    end_big = timer()
    print(timedelta(seconds=end_big-start_big))

# For RigidReg resulting fields using affine + SyN on rigid pre-reg images
def AffineSynTransfoPair1_5():

    p_dict = PatientToScanID()

    times = []
    mov_fix_s = []

    start_big = timer()
    for p in p_dict:
        list_scanIDs = sorted(os.listdir(f"/media/andjela/SeagatePor/PairRegData/images/{p}/"))
        intra_sample_indices = []
        num_images_in_group = len(list_scanIDs)
        for i in range(num_images_in_group):
            for j in range(i):
                intra_sample_indices.append((j, i))
        num_pairs = len(intra_sample_indices)
        #Iteration over possible pairs for specific patient
        #access index of first image to register with intra_sample_indices[0][0] for first pair possible
        for i in range(num_pairs):
            moving = list_scanIDs[intra_sample_indices[i][0]] #gives specific scanID for moving image
            fixed = list_scanIDs[intra_sample_indices[i][1]] #gives specific scanID for fixed image
            if not os.path.exists(f'/media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/'): #[:-7] to remove .nii.gz from
                os.mkdir(f'/media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/')
            start = time.time()

            # 850x250x250,1e-7,25 (replace with this convergence?)
            os.system(f"antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 " \
                f"--initial-moving-transform [ /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}.nii.gz, /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}.nii.gz, 0 ] " \
                f"--initialize-transforms-per-stage 0 " \
                f"--interpolation Linear " \
                f"--output [ /media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_, /media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz, /media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/fix2mov_warped_image.nii.gz ] " \
                f"--transform Affine[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}.nii.gz, /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--transform SyN[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}.nii.gz, /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--write-composite-transform 1 " \
                f"--verbose 1 " 
                )

            end = time.time()
            # DDF creation from the h5 file
            os.system(f"antsApplyTransforms -d 3 -o [/media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/ddf_ANTs.nii.gz, 1] -v 1 -t /media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_Composite.h5 " \
                f"-r  /media/andjela/SeagatePor/RigidReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}")

            mov_fix_s.append(f'{moving[:-7]}_{fixed[:-7]}')
            times.append(end-start)

    time_taken = {'scanIDs' : mov_fix_s ,  'times': times}
    df_timer = pd.DataFrame(data=time_taken)


    df_timer.to_csv(f"/media/andjela/SeagatePor/PairReg/affine-syn-1.5/images/time_taken")

    end_big = timer()
    print(timedelta(seconds=end_big-start_big))

# For RigidAffineReg resulting fields
def SynTransfoPair1_5():

    p_dict = PatientToScanID()

    times = []
    mov_fix_s = []

    start_big = timer()
    for p in p_dict:
        list_scanIDs = sorted(os.listdir(f"/media/andjela/SeagatePor/PairRegData/images/{p}/"))
        intra_sample_indices = []
        num_images_in_group = len(list_scanIDs)
        for i in range(num_images_in_group):
            for j in range(i):
                intra_sample_indices.append((j, i))
        num_pairs = len(intra_sample_indices)
        #Iteration over possible pairs for specific patient
        #access index of first image to register with intra_sample_indices[0][0] for first pair possible
        for i in range(num_pairs):
            moving = list_scanIDs[intra_sample_indices[i][0]] #gives specific scanID for moving image
            fixed = list_scanIDs[intra_sample_indices[i][1]] #gives specific scanID for fixed image
            if not os.path.exists(f'/media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/'): #[:-7] to remove .nii.gz from
                os.mkdir(f'/media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/')
            start = time.time()
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ /media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_, /media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz, /media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform SyN[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/RigidAffineReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}.nii.gz, /media/andjela/SeagatePor/RigidAffineReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox "
                )
            end = time.time()

            # DDF creation from the h5 file
            os.system(f"antsApplyTransforms -d 3 -o [/media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/ddf_ANTs.nii.gz, 1] -v 1 -t /media/andjela/SeagatePor/PairReg/syn-1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_Composite.h5 " \
                f"-r  /media/andjela/SeagatePor/RigidAffineReg_1.5/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed}")
            
            mov_fix_s.append(f'{moving[:-7]}_{fixed[:-7]}')
            times.append(end-start)

    time_taken = {'scanIDs' : mov_fix_s ,  'times': times}
    df_timer = pd.DataFrame(data=time_taken)


    df_timer.to_csv(f"/media/andjela/SeagatePor/PairReg/syn-1.5/images/time_taken")

    end_big = timer()
    print(timedelta(seconds=end_big-start_big))

# Rigid pre-alignment
def RigidTransfoPair():

    p_dict = PatientToScanID()

    start = timer()
    for p in p_dict:
        list_scanIDs = sorted(os.listdir(f"/media/andjela/SeagatePor/PairRegData/images/{p}/"))
        intra_sample_indices = []
        num_images_in_group = len(list_scanIDs)
        for i in range(num_images_in_group):
            for j in range(i):
                intra_sample_indices.append((j, i))
        num_pairs = len(intra_sample_indices)
        #Iteration over possible pairs for specific patient
        #access index of first image to register with intra_sample_indices[0][0] for first pair possible
        for i in range(num_pairs):
            moving = list_scanIDs[intra_sample_indices[i][0]] #gives specific scanID for moving image
            fixed = list_scanIDs[intra_sample_indices[i][1]] #gives specific scanID for fixed image
            if not os.path.exists(f'/media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/'): #[:-7] to remove .nii.gz from
                os.mkdir(f'/media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/')
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_, /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz, /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed}, /media/andjela/SeagatePor/PairRegData/images/{p}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox "
                )
            # warp subject calculated moving mask to fixed
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input /home/andjela/Documents/work_dir2/cbf2mni_wdir/{p}/{moving[:-7]}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation NearestNeighbor --output /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mask_trans.nii " \
                f"--reference-image /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed} " \
                f"--transform /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_Composite.h5")
            #brain extraction for moving image which becomes mov2fix_warped
            os.system(f'fslmaths /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz -mul /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/mask_trans.nii /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}_dtype.nii.gz -odt float')
            #brain extraction for fixed image which stays fixed
            os.system(f'fslmaths /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed} -mul /home/andjela/Documents/work_dir2/cbf2mni_wdir/{p}/{fixed[:-7]}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz /media/andjela/SeagatePor/PairReg/rigid/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}_dtype.nii.gz -odt float')
    end = timer()
    print(timedelta(seconds=end-start))

# Rigid+affine pre-alignment
def RigidAffineTransfoPair():

    p_dict = PatientToScanID()

    start = timer()
    for p in p_dict:
        list_scanIDs = sorted(os.listdir(f"/media/andjela/SeagatePor/PairRegData/images/{p}/"))
        intra_sample_indices = []
        num_images_in_group = len(list_scanIDs)
        for i in range(num_images_in_group):
            for j in range(i):
                intra_sample_indices.append((j, i))
        num_pairs = len(intra_sample_indices)
        #Iteration over possible pairs for specific patient
        #access index of first image to register with intra_sample_indices[0][0] for first pair possible
        for i in range(num_pairs):
            moving = list_scanIDs[intra_sample_indices[i][0]] #gives specific scanID for moving image
            fixed = list_scanIDs[intra_sample_indices[i][1]] #gives specific scanID for fixed image
            if not os.path.exists(f'/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/'): #[:-7] to remove .nii.gz from
                os.mkdir(f'/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/')
            os.system(f"antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 " \
                f"--initial-moving-transform [ /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed}, /media/andjela/SeagatePor/PairRegData/images/{p}/{moving}, 1 ] " \
                f"--initialize-transforms-per-stage 0 " \
                f"--interpolation Linear " \
                f"--output [ /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_, /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz, /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/fix2mov_warped_image.nii.gz ] " \
                f"--transform Rigid[ 0.1 ] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed}, /media/andjela/SeagatePor/PairRegData/images/{p}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--transform Affine[ 0.1 ] " \
                f"--metric Mattes[ /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed}, /media/andjela/SeagatePor/PairRegData/images/{p}/{moving}, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-estimate-learning-rate-once 1 --use-histogram-matching 1 " \
                f"--winsorize-image-intensities [ 0.005, 0.995 ] " \
                f"--write-composite-transform 1" \
                f"--verbose 1"
                )
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{moving[:-7]}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation Linear " \
                f"--output /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mask_trans.nii " \
                f"--reference-image /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed} " \
                f"--transform /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_Composite.h5"
                )
            #binarize mask
            os.system(f'fslmaths /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mask_trans.nii -bin /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/masknew_trans.nii -odt float')
            #brain extraction for moving image which becomes mov2fix_warped
            os.system(f'fslmaths /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/mov2fix_warped_image.nii.gz -mul /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/masknew_trans.nii /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/{moving[:-7]}_dtype.nii.gz -odt float')
            #brain extraction for fixed image which stays fixed
            os.system(f'fslmaths /media/andjela/SeagatePor/PairRegData/images/{p}/{fixed} -mul /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{fixed[:-7]}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz /media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{moving[:-7]}_{fixed[:-7]}/{fixed[:-7]}_dtype.nii.gz -odt float')
    end = timer()
    print(timedelta(seconds=end-start))

def GroupedCrossValidationDataPair(PROJECT_DIR,label, NEW_DIR):

    os.chdir(PROJECT_DIR)
    DATA_PATH = "images"

    images_path = os.path.join(PROJECT_DIR, DATA_PATH)

    if label == True:
        LABEL_PATH = "labels"
        labels_path = os.path.join(PROJECT_DIR, LABEL_PATH)

    all_pairs = np.array(sorted(os.listdir(images_path)))
    
    patient_id = [name[:5] for name in all_pairs]
   
    all_patients = sorted(list(set(patient_id)))

    # Modify patient_id to be 0 to 63 instead of 10006 to 10163
    dict = {}
    for i, value in enumerate(all_patients):
        dict[value]=i
    modif_patient_id = []
    for elem in patient_id:
        for key, value in dict.items():
            if elem == key:
                elem = value
                modif_patient_id.append(elem)
    
    group_kfold = GroupKFold(n_splits=5)
    n_splits = group_kfold.get_n_splits(X=all_pairs, groups=modif_patient_id)

    # Iterate through folds
    i = 0 
    for train_index, test_index in group_kfold.split(X=all_pairs, groups=modif_patient_id):
        X_train, X_test = all_pairs[train_index], all_pairs[test_index]
        # Rename for each fold
        
        DIR = NEW_DIR + f'_f{i}'
        folders = [os.path.join(DIR, dn) for dn in ["train", "test"]]
        os.mkdir(DIR)
        for fn in folders:
            os.mkdir(fn)
            os.mkdir(os.path.join(fn, "images"))
            if label == True:
                os.mkdir(os.path.join(fn, "labels"))

        for p in X_test:
            if not os.path.exists(os.path.join(DIR, "test", "images",p)):
                os.mkdir(os.path.join(DIR, "test", "images",p))
            copy_tree(os.path.join(images_path, p), os.path.join(DIR, "test", "images",p))
            if label == True:
                if not os.path.exists(os.path.join(DIR, "test", "labels",p)):
                    os.mkdir(os.path.join(DIR, "test", "labels",p))
                copy_tree(os.path.join(labels_path, p), os.path.join(DIR, "test", "labels",p))

        for p in X_train:
            if not os.path.exists(os.path.join(DIR, "train", "images",p)):
                os.mkdir(os.path.join(DIR, "train", "images",p))
            copy_tree(os.path.join(images_path, p), os.path.join(DIR, "train", "images",p))
            if label == True:
                if not os.path.exists(os.path.join(DIR, "train", "labels",p)):
                    os.mkdir(os.path.join(DIR, "train", "labels",p))
                copy_tree(os.path.join(labels_path, p), os.path.join(DIR, "train", "labels",p))
        i+=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                    FUNCTIONS FOR PAIR_BASED_REG                                                                   #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # # ANTs Comparison Functions below
    # # For NoReg
    # RigidAffineSynTransfoPair1_5()
    # SynOnlyTransfoPair1_5()
    # # For RigidReg
    # AffineSynTransfoPair1_5()
    # SynTransfoPairForRigid1_5()
    # # For RigidAffineReg
    # SynTransfoPair1_5()

    # # Pre-alignments for DL-based approaches
    # RigidTransfoPair()
    # RigidAffineTransfoPair()

    # # 5-fold cross-validation
    # label=True
    # res = '1.5'

    # PROJECT_DIR = f"/media/andjela/SeagatePor/NoReg_{res}"
    # NEW_DIR = f"/media/andjela/SeagatePor/dataset_NoReg_{res}"

    # PROJECT_DIR = f"/media/andjela/SeagatePor/RigidReg_{res}"
    # NEW_DIR = f"/media/andjela/SeagatePor/dataset_RigidReg_{res}"

    # PROJECT_DIR = f"/media/andjela/SeagatePor/RigidAffineReg_{res}"
    # NEW_DIR = f"/media/andjela/SeagatePor/dataset_RigidAffineReg_{res}"

    # GroupedCrossValidationDataPair(PROJECT_DIR,label, NEW_DIR)