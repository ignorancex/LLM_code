import shutil
import json
import pandas
import nibabel as nib
import numpy
from shutil import copyfile
from datetime import datetime
from xnat2mids.procedures import Procedures
import SimpleITK as sitk
body_part_bids = ['head', 'brain', 'skull']

class ProceduresMR(Procedures):
    def __init__(self):
        self.reset_indexes()

    def reset_indexes(self):
        """Reset all values in dictionary of the atribute "dict_indexes"."""

        self.run_dict = {}


    def control_sequences(
        self, folder_nifti, mids_session_path, session_name, dict_json, protocol, acq, dir_, folder_BIDS, acquisition_date_time_correct, body_part
    ):
        
        folder_image_mids = mids_session_path.joinpath(
             "" if body_part.lower() in body_part_bids else "mim-mr",
            folder_BIDS
        )
        
        #folder_image_mids.mkdir(parents=True, exist_ok=True)



        self.control_image(folder_nifti, folder_image_mids, session_name, dict_json, protocol, acq, dir_,acquisition_date_time_correct, body_part)

    def control_image(self, folder_conversion, folder_image_mids, session_name, dict_json, protocol, acq, dir_, acquisition_date_time_correct, body_part):

        """

        """


        # Search all nifti files in the old folder and sort them
        nifti_files = sorted([i for i in folder_conversion.glob("*.nii.gz")])

        len_nifti_files = len(nifti_files)
        if len_nifti_files == 0: return
        folder_image_mids.mkdir(parents=True, exist_ok=True)
        # This is a counter for several nifties in one adquisition

        protocol_label = f'{protocol}'
        acq_label = f'{acq}' if acq else ''
        bp_label = f"{body_part}"
        vp_label = [
            (
                f"{self.get_plane_nib(nifti_file)}"
                if body_part.lower() in body_part_bids
                else f"{self.get_plane_nib(nifti_file)}"
            )
            for nifti_file in nifti_files
        ]
        
        
        key = json.dumps([session_name, acq_label, dir_, bp_label, vp_label, protocol_label])
        value = self.run_dict.get(key, [])
        value.append({
            "run":nifti_files, 
            "series_number": dict_json.get("SeriesNumber", ), 
            "adquisition_time":datetime.fromisoformat(acquisition_date_time_correct),
            "folder_mids": folder_image_mids}
            )
        
        self.run_dict[key] = value

    def copy_sessions(self, subject_name):
        for key, runs_list in self.run_dict.items():
            df_aux = pandas.DataFrame.from_dict(runs_list)
            df_aux.sort_values(by="adquisition_time", inplace = True)
            df_aux.index = numpy.arange(1, len(df_aux) + 1)
            activate_run = True #if len(df_aux) > 1 else False
            for index, row in df_aux.iterrows():

                activate_chunk_partioned = True if len(row['run']) > 1 else False
                print("-"*79)
                for acq, file_ in enumerate(sorted(row['run'])):

                    dest_file_name = self.calculate_name(
                        subject_name=subject_name, 
                        keys=key,
                        num_run=row["series_number"], 
                        num_part=acq, 
                        activate_run=activate_run, 
                        activate_chunk_partioned=activate_chunk_partioned
                    )
                    
                    print("origen:", file_)
                    print("destino:", row["folder_mids"].joinpath(str(dest_file_name) + "".join(file_.suffix)))
                    
                    row["folder_mids"].mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file_, row["folder_mids"].joinpath(str(dest_file_name) + "".join(file_.suffixes)))
                other_files = [f for f in file_.parent.iterdir() if file_.suffix not in str(f) and not f.is_dir()]
                for other_file in other_files:
                    print("origen:", other_file)
                    print("destino:", row["folder_mids"].joinpath(str(dest_file_name) + "".join(other_file.suffixes)))
                    shutil.copyfile(str(other_file), row["folder_mids"].joinpath(str(dest_file_name) + "".join(other_file.suffixes)))
                print("-"*79)
    def get_plane_nib(self, nifti):
        """
            Calculate the type of plane with the tag image orientation patient
            in dicom metadata.
        """

        img = nib.load(nifti)
        plane = nib.aff2axcodes(img.affine)[2]
        return "ax" if plane in ["S", "I"] else "sag" if plane in ["R", "L"] else "cor"

    def calculate_name(self, subject_name, keys, num_run, num_part, activate_run, activate_chunk_partioned):

        key_list = json.loads(keys)
        sub = subject_name
        ses = key_list[0]
        acq = f"acq-{key_list[1]}" if key_list[1] else ''
        dir_ = f"dir-{key_list[2]}" if key_list[2] else ''
        run = f"run-{num_run}" if activate_run else ''
        chunk = f"chunk-{num_part+1}" if activate_chunk_partioned else ''
        bp = '' if key_list[3].lower() in body_part_bids else f'bp-{key_list[3].lower()}'
        lat = ''
        vp = '' if key_list[3].lower() in body_part_bids else f'vp-{key_list[4][num_part-1]}'
        mod = key_list[5]
        
        return "_".join([
            part for part in [
                sub, ses, acq, dir_, run, chunk, bp, lat, vp, mod
            ] if part != ''
        ])
