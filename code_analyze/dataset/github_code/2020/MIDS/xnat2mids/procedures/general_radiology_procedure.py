import pandas
import numpy
import json
import shutil
from datetime import datetime
from xnat2mids.procedures import Procedures
from pathlib import Path

view_position_regex = r"pa|ap|lat"
class RadiologyProcedure(Procedures):
    def __init__(self):
        self.view_positions_2d = ['pa', 'lat', 'ap', None]
        self.view_positions_3d = ['ax', 'sag', 'cor', None]
        self.reset_indexes()

    # def reset_indexes(self):
    #     self.dict_indexes = {
    #         "cr": {k: 1 for k in self.view_positions_2d},
    #         "dx": {k: 1 for k in self.view_positions_2d},
    #         "ct": {k: 1 for k in self.view_positions_3d},
    #         # not defined yet
    #         "bmd": {None: 1},
    #         "xa": {None: 1},
    #         "io": {None: 1},
    #         "mg": {None: 1},
    #         "vf": {None: 1},
    #         "rf": {None: 1},
    #         "rtimage": {None: 1},
    #         "rtplan": {None: 1},
    #         "rtrecord": {None: 1},
    #         "rtdose": {None: 1},
    #         "df": {None: 1},
    #         "rg": {None: 1},
    #         "ds": {None: 1},
    #     }
    # def __init__(self):
    #         self.reset_indexes()

    def reset_indexes(self):
        self.run_dict = {}

    def control_image(
                        self, 
                        folder_conversion, 
                        mids_session_path,
                        dict_json, 
                        session_name, 
                        modality_,
                        laterality,
                        acquisition_date_time, 
                        body_part
                    ):
        png_files = [i for i in folder_conversion.glob("*.png")]
        #nifti_files = sorted([i for i in folder_conversion.glob("*.nii*")])
        #mim = modality if body_part.lower in[ "head", "brain"] else "mim-rx"
        laterality = dict_json.get("Laterality")
        view_position = dict_json.get("ViewPosition")
        len_files = len(png_files)
        if not len_files: return

        key = json.dumps([session_name, body_part, laterality, view_position, modality_])
        value = self.run_dict.get(key, [])
        value.append({
            "run":png_files,
            "series_number": dict_json.get("SeriesNumber"),
            "adquisition_time":datetime.fromisoformat(acquisition_date_time), 
            "folder_mids": mids_session_path})
        self.run_dict[key]=value
        
        

    def copy_sessions(self, subject_name):
        for key, runs_list in self.run_dict.items():
            df_aux = pandas.DataFrame.from_dict(runs_list)
            df_aux.sort_values(by="adquisition_time", inplace = True)
            df_aux.index = numpy.arange(1, len(df_aux) + 1)
            print(len(df_aux))
            activate_run = True #if len(df_aux) > 1 else False
            print(f"{activate_run}")
            for index, row in df_aux.iterrows():
                activate_chunk_partioned = True if len(row['run']) > 1 else False
                for acq, file_ in enumerate(sorted(row['run'])):
                
                    dest_file_name = self.calculate_name(
                        subject_name=subject_name, 
                        key=key,
                        num_run=row["series_number"], 
                        num_part=acq, 
                        activate_run=activate_run, 
                        activate_chunk_partioned=activate_chunk_partioned
                    )
                    print("-"*79)
                    print(row["folder_mids"])
                    print("-"*79)
                    print("origen:", file_)
                    print("destino:", row["folder_mids"].joinpath(str(dest_file_name) + "".join(file_.suffix)))
                    
                    row["folder_mids"].mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file_, row["folder_mids"].joinpath(str(dest_file_name) + "".join(file_.suffix)))
                other_files = [f for f in file_.parent.iterdir() if file_.suffix not in str(f) and not f.is_dir()]
                for other_file in other_files:
                    print("origen:", other_file)
                    print("destino:", row["folder_mids"].joinpath(str(dest_file_name) + "".join(other_file.suffixes)))
                    shutil.copyfile(str(other_file), row["folder_mids"].joinpath(str(dest_file_name) + "".join(other_file.suffixes)))

    def calculate_name(self, subject_name, key, num_run, num_part, activate_run, activate_chunk_partioned):
        key_list = json.loads(key)
        print(key_list)
        # print(num_part, activate_chunk_partioned)
        sub = subject_name
        ses = key_list[0]
        #rec = f"rec-{key_list[1]}" if key_list[1] else ''
        run = f"run-{num_run}" if activate_run else ''
        bp = f"bp-{key_list[1].lower()}" if key_list[1] else ''
        lat = f"lat-{key_list[2].lower()}" if key_list[2] else ''
        vp = f"vp-{key_list[3].lower()}" if key_list[3] else ''
        chunk = f"chunk-{num_part+1}" if activate_chunk_partioned else ''
        mod = key_list[4].lower()
        print(f"{key=}")
        return "_".join([
            part for part in [
                sub, ses, run, bp, lat, vp, chunk, mod
            ] if part != ''
        ])
   