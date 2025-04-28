import pandas as pd 
import nibabel 
import numpy as np
import uuid
import glob
import os

from tqdm import tqdm

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


cfg = {
        "smax_l": {"x":    { 
                            "mean":57.6290705679648,
                            "std":1.109474475815243
                            },

                 "y":  { 
                            "mean":174.74581890830794,
                            "std":1.184642904082489
                            },

                 "z":      { 
                            "mean":91.19085451370563,
                            "std":1.3316953801602163
                            },
                 },

        "smax_r": {"x":  { 
                            "mean":112.07762085692738,
                            "std":1.4333812140578561
                            },

                 "y":  { 
                            "mean":174.68885602937354,
                            "std":1.259285246719773
                            },

                 "z":      { 
                            "mean":88.92680928824241,
                            "std":4.872207337495459
                            },
                 },

        "nose": {"x":  { 
                            "mean":83.92852228020257,
                            "std":1.2969000921858846
                            },

                 "y":  { 
                            "mean":174.50147322273253,
                            "std":2.548013413953404
                            },

                 "z":     { 
                            "mean":98.49876853271435,
                            "std":2.2851872407946523
                            },
                 },
        
      }


def __get__crop__(data,location,total_samples=20,std_factor=1,size_of_crop=40,flip=False):
    
    #Function to crop out sub volume
    x,y,z =  int(location["x"]["mean"]),int(location["y"]["mean"]),int(location["z"]["mean"])
    x_std,y_std,z_std =  int(location["x"]["std"]),int(location["y"]["std"]),int(location["z"]["std"])

    arr_x = np.random.normal(x, x_std*std_factor, total_samples)    
    arr_y = np.random.normal(y, y_std*std_factor, total_samples)    
    arr_z = np.random.normal(z, z_std*std_factor, total_samples)    
    
    all_samples = []
    for x,y,z in zip(arr_x,arr_y,arr_z):
        
        x_start = x - size_of_crop//2
        x_end   = x_start + size_of_crop
        y_start = y - size_of_crop//2
        y_end   = y_start + size_of_crop
        z_start = z - size_of_crop//2
        z_end   = z_start + size_of_crop

        out = data[int(x_start):int(x_end),int(y_start):int(y_end),int(z_start):int(z_end)]
        
        if flip:
            out =  np.array(np.flip(out, axis=0), dtype=np.float)
        
        all_samples.append(out)
    
    return all_samples


for size_of_crop in [65]: #[15,20,25,30,35,40,45,50,55]
    
    for std_factor in [1]: #[1,2,3]:

        #std_factor = 1
        total_samples = 15


        mri_root = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/all_registered_ds/{}.nii.gz"
        root_save_path =  f"/home/debayan/Desktop/MRI_HCHS/JAMA-unlabelled-1000-corrected/crop_size_{size_of_crop}_std_factor_{std_factor}/"
        root_split_path =  f"/home/debayan/Desktop/MRI_HCHS/JAMA-unlabelled-1000-corrected/splits/"

        create_dir(root_save_path)
        create_dir(root_split_path)


        df_label = pd.read_excel('//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/03.04.2023-labelled.xlsx',engine='openpyxl', index_col=False)
        df_label = df_label.fillna('NA')
        #print(df_label)
        df_label['MRI-quality'] = df_label['MRI-quality'].astype(str)
        df_label = df_label[df_label['Proband']!="NA"]  
        #df_label = df_label[df_label['MRI-quality']=="NA"] 
        df_label = df_label.loc[:, ['Proband','Kieferhöhle']]
        df_label['Proband'] = df_label['Proband'].astype(str)


        df_unlabelled = pd.read_excel('//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/02.03.2023-unlabelled.xlsx',engine='openpyxl', index_col=False)
        df_unlabelled = df_unlabelled.fillna('NA')
        df_unlabelled = df_unlabelled[df_unlabelled['Proband']!="NA"]  
        #df_unlabelled['MRI-quality'] = df_unlabelled['MRI-quality'].astype(str)
        #df_unlabelled = df_unlabelled.loc[:, ['Proband','Kieferhöhle']]



        print("total labelled MRIs",len(df_label))
        print("total unlabelled MRIs",len(df_unlabelled))
        #print(df)
        classes = {"1":"mucousa","2": "polyp","3":"Cyst","5":"Fully Occupied"}

        occurrance = {
                        "1":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},
                        "2":{"smax":0,"seth":0,"smax_l":0,"smax_r":0,"sfront":0,"sphen":0},
                        "3":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},
                        "4":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},
                        "5":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},
                        "no_path":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0}
                    }

        patient_list_normal = []
        patient_list_bad = []

        patient_list = {"patient_id":[],"smax_l":[],"smax_r":[]}
        save_path_locations  = []


        
        for df in [df_unlabelled]:

            for index, row in tqdm(df.iterrows(),total=len(df)):
                patient_id = row["Proband"].split(" ")[0]

                try:
                    img = nibabel.load(mri_root.format(patient_id.lower()))
                    
                    save_path_smax_l = root_save_path+f"/{patient_id}/smax_l"
                    save_path_locations.append(save_path_smax_l)
                    create_dir(save_path_smax_l)
                    save_path_smax_r = root_save_path+f"/{patient_id}/smax_r"
                    save_path_locations.append(save_path_smax_r)
                    create_dir(save_path_smax_r)
                
                except:

                    print("Not found Normal",patient_id)
                    continue
                

                roi_objects = __get__crop__(img.get_data(),cfg["smax_l"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop)
                for index,roi in enumerate(roi_objects):
                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                    nibabel.save(new_image, save_path_smax_l + f"/{index}_" + "smax_l.nii.gz")

                roi_objects = __get__crop__(img.get_data(),cfg["smax_r"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop,flip=True)
                for index,roi in enumerate(roi_objects):
                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                    nibabel.save(new_image, save_path_smax_r + f"/{index}_" + "smax_r.nii.gz")

        df = pd.DataFrame.from_dict({"folder":save_path_locations,"label":[0]*len(save_path_locations)})
        df.to_csv(root_split_path + f"split_cc_{size_of_crop}.txt")
        
        
        for df in [df_label]:

            patient_list_bad = []
            patient_list_normal = []
            for index, row in tqdm(df.iterrows(),total=len(df)):
                
                patient_id = row["Proband"].split(" ")[0]

                if row["Kieferhöhle"] == "NA":

                    #Normal
                    try:

                        img = nibabel.load(mri_root.format(patient_id.lower()))
                        patient_list_normal.append(patient_id)
                        
                        save_path_l = root_save_path+f"/{patient_id}/normal-l/"
                        save_path_r = root_save_path+f"/{patient_id}/normal-r/"
                        create_dir(save_path_l)
                        create_dir(save_path_r)
                        
                    except:

                        print("Not found Normal",patient_id)
                        continue
                    
                    #extracting left maxillary sinus volumes
                    roi_objects = __get__crop__(img.get_data(),cfg["smax_l"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop)

                    for index,roi in enumerate(roi_objects):
                        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                        nibabel.save(new_image, save_path_l + f"/{index}_" + "smax_l.nii.gz")

                    #extracting right maxillary sinus volumes
                    roi_objects = __get__crop__(img.get_data(),cfg["smax_r"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop,flip=True)
                    for index,roi in enumerate(roi_objects):
                        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                        nibabel.save(new_image, save_path_r + f"/{index}_" + "smax_r.nii.gz")
                    
                    occurrance["no_path"]["smax_r"] +=  1
                    occurrance["no_path"]["smax_l"] +=  1
                        

                else:

                    #Abnormal
                    try:

                        img = nibabel.load(mri_root.format(patient_id.lower()))
                        #patient_list_bad.append(patient_id)

                    except:

                        print("Not found Normal",patient_id)
                        continue

                    categories = row["Kieferhöhle"].split(",")
                    right_bad = False
                    left_bad = False

                    for cat in categories:
                        
                        if "l" in cat:

                            left_bad = True
                        
                            #extracting left maxillary sinus volumes
                            roi_objects = __get__crop__(img.get_data(),cfg["smax_l"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop)

                            if "1" in cat: 
                                occurrance["1"]["smax_l"] +=  1
                                save_path = root_save_path+f"/{patient_id}/mucosal_thickening-l/"
                                
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                for index,roi in enumerate(roi_objects):
                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "2" in cat: 
                                save_path = root_save_path+f"/{patient_id}/polyps-l/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["2"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "3" in cat: 
                                save_path = root_save_path+f"/{patient_id}/cysts-l/"
                                create_dir(save_path)

                                patient_list_bad.append(patient_id)
                                occurrance["3"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "5" in cat: 
                                save_path = root_save_path+f"/{patient_id}/fully_occupied-l/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["5"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")
                            
                            
                            #nibabel.save(new_image,roi_dataset.format("bad",patient_id + "_"+str(uuid.uuid1())))

                        elif "r" in cat:
                            
                            right_bad = True
                        
                            #extracting left maxillary sinus volumes
                            roi_objects = __get__crop__(img.get_data(),cfg["smax_r"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop,flip=True)

                            if "1" in cat: 
                                occurrance["1"]["smax_r"] +=  1
                                save_path = root_save_path+f"/{patient_id}/mucosal_thickening-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                for index,roi in enumerate(roi_objects):
                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                            if "2" in cat: 
                                save_path = root_save_path+f"/{patient_id}/polyps-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["2"]["smax_r"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                            if "3" in cat: 
                                save_path = root_save_path+f"/{patient_id}/cysts-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["3"]["smax_r"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")


                            if "5" in cat: 
                                save_path = root_save_path+f"/{patient_id}/fully_occupied-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["5"]["smax_r"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")
                            

                        elif "b" in cat: 

                            left_bad = True
                        
                            #extracting left maxillary sinus volumes
                            roi_objects = __get__crop__(img.get_data(),cfg["smax_l"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop)

                            if "1" in cat: 
                                occurrance["1"]["smax_l"] +=  1
                                save_path = root_save_path+f"/{patient_id}/mucosal_thickening-l/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                for index,roi in enumerate(roi_objects):
                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "2" in cat: 
                                save_path = root_save_path+f"/{patient_id}/polyps-l/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["2"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "3" in cat: 
                                save_path = root_save_path+f"/{patient_id}/cysts-l/"
                                create_dir(save_path)

                                patient_list_bad.append(patient_id)
                                occurrance["3"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            if "5" in cat: 
                                save_path = root_save_path+f"/{patient_id}/fully_occupied-l/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["5"]["smax_l"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")

                            right_bad = True
                        
                            #extracting right maxillary sinus volumes
                            roi_objects = __get__crop__(img.get_data(),cfg["smax_r"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop,flip=True)

                            if "1" in cat: 
                                occurrance["1"]["smax_r"] +=  1
                                save_path = root_save_path+f"/{patient_id}/mucosal_thickening-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)

                                for index,roi in enumerate(roi_objects):
                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                            if "2" in cat: 
                                save_path = root_save_path+f"/{patient_id}/polyps-r/"
                                create_dir(save_path)
                                occurrance["2"]["smax_r"] +=  1
                                patient_list_bad.append(patient_id)

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                            if "3" in cat: 
                                save_path = root_save_path+f"/{patient_id}/cysts-r/"
                                create_dir(save_path)
                                occurrance["3"]["smax_r"] +=  1
                                patient_list_bad.append(patient_id)

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                            if "5" in cat: 
                                save_path = root_save_path+f"/{patient_id}/fully_occupied-r/"
                                create_dir(save_path)
                                patient_list_bad.append(patient_id)
                                occurrance["5"]["smax_r"] +=  1

                                for index,roi in enumerate(roi_objects):

                                    new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                                    nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")

                    if not right_bad:

                        save_path = root_save_path+f"/{patient_id}/normal-r/"
                        create_dir(save_path)
                        #extracting right maxillary sinus volumes
                        roi_objects = __get__crop__(img.get_data(),cfg["smax_r"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop,flip=True)
                        occurrance["no_path"]["smax_r"] +=  1

                        for index,roi in enumerate(roi_objects):

                            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                            nibabel.save(new_image, save_path + f"/{index}_" + "smax_r.nii.gz")
                        

                    if not left_bad: 

                        save_path = root_save_path+f"/{patient_id}/normal-l/"
                        create_dir(save_path)
                        #extracting right maxillary sinus volumes
                        roi_objects = __get__crop__(img.get_data(),cfg["smax_l"],total_samples=total_samples,std_factor=std_factor,size_of_crop=size_of_crop)
                        occurrance["no_path"]["smax_l"] +=  1

                        for index,roi in enumerate(roi_objects):

                            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))                        
                            nibabel.save(new_image, save_path + f"/{index}_" + "smax_l.nii.gz")


        print("Statistics",occurrance)
        print("Normal",len(list(set(patient_list_normal))))
        print("Bad",len(list(set(patient_list_bad))))


        





                





