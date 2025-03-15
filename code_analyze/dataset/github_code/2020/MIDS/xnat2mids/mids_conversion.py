import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas
from xnat2mids.conversion.io_json import load_json
from xnat2mids.conversion.dicom_converters import generate_json_dicom
from xnat2mids.procedures.magnetic_resonance_procedures import ProceduresMR
from xnat2mids.procedures.light_procedures import LightProcedure
from xnat2mids.procedures.general_radiology_procedure import RadiologyProcedure
from xnat2mids.protocols.scans_tagger import Tagger
from xnat2mids.conversion.dicom_converters import dicom2nifti
from xnat2mids.conversion.dicom_converters import dicom2png
from tqdm import tqdm
from pandas.errors import EmptyDataError
import logging
##1003-2-4T02:23:43.3245
##20231212020401.23452
adquisition_date_pattern_2 = r"(?P<fecha1>(?P<year>\d{4})-(?P<month>\d+)-(?P<day>\d+)T(?P<hour>\d+):(?P<minutes>\d+):(?P<seconds>\d+).(?P<ms>\d+))|(?P<fecha2>(?P<year2>\d{4})(?P<month2>\d{2})(?P<day2>\d{2})(?P<hour2>\d{2})(?P<minutes2>\d{2})(?P<seconds2>\d{2}).(?P<ms2>\d+))"
adquisition_date_pattern = r"(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)T(?P<hour>\d+):(?P<minutes>\d+):(?P<seconds>\d+).(?P<ms>\d+)"
subses_pattern = r"[A-z]+(?P<prefix_sub>\d*)?(_S)(?P<suffix_sub>\d+)(\\|/)[A-z]+\-?[A-z]*(?P<prefix_ses>\d*)?(_E)(?P<suffix_ses>\d+)"
prostate_pattern = r"(?:(?:(?:diff?|dwi)(?:\W|_)(?:.*)(?:b\d+))|dif 1500)|stir|(?:adc|Apparent)|prop|blade|fse|tse|^ax T2$"
chunk_pattern = r"_chunk-(?P<chunk>\d)+"

aquisition_date_pattern_comp = re.compile(adquisition_date_pattern_2)
prostate_pattern_comp = re.compile(prostate_pattern, re.I)
chunk_pattern_comp = re.compile(chunk_pattern)
dict_keys = {
    'Modality': '00080060',
    'SeriesDescription': '0008103E',
    'ProtocolName': '00181030',
    'ComplexImage Component Attribute': '00089208',
    "ImageType" :'00080008',
    #"difusion Directionality": ''
}

dict_mr_keys = {
    'Manufacturer': '00080070',
    'ManufacturerModelName': '00081090',
    'ScanningSequence': '00180020',
    'SequenceVariant': '00180021',
    'ScanOptions': '00180022',
    'ImageType': '00080008',
    'AngioFlag': '00180025',
    'MagneticFieldStrength': '00180087',
    'RepetitionTime': '00180080',
    'InversionTime': '00180082',
    'FlipAngle': '00181314',
    'EchoTime': '00180081',
    'SliceThickness': '00180050',
    'SeriesDescription': '0008103E',
}

BIOFACE_PROTOCOL_NAMES = [
    '3D-T2-FLAIR SAG',
    '3D-T2-FLAIR SAG NUEVO-1',
    #'AAhead_scout',
    'ADVANCED_ASL',
    'AXIAL T2 TSE FS',
    'AX_T2_STAR',
    'DTIep2d_diff_mddw_48dir_p3_AP', #
    'DTIep2d_diff_mddw_4b0_PA', #
    'EPAD-3D-SWI',
    'EPAD-B0-RevPE', # PA
    'EPAD-SE-fMRI',
    'EPAD-SE-fMRI-RevPE',
    'EPAD-SingleShell-DTI48', # AP
    'EPAD-rsfMRI (Eyes Open)',
    'MPRAGE_GRAPPA2', # T1 mprage
    'asl_3d_tra_iso_3.0_highres',
    'pd+t2_tse_tra_p2_3mm',
    't1_mprage_sag_p2_iso', # t1
    't2_space_dark-fluid_sag_p2_iso', # flair
    't2_swi_tra_p2_384_2mm'
]

BIOFACE_PROTOCOL_NAMES_DESCARTED = [
    #'DTIep2d_diff_mddw_48dir_p3_AP',
    #'DTIep2d_diff_mddw_4b0_PA',
    'EPAD-B0-RevPE',
    'EPAD-SingleShell-DTI48',
    'EPAD-3D-SWI',
    'EPAD-SE-fMRI',
    'EPAD-rsfMRI (Eyes Open)',
    'EPAD-SE-fMRI-RevPE',
    'AAhead_scout',
    'ADVANCED_ASL',
    'MPRAGE_GRAPPA2',
    '3D-T2-FLAIR SAG',
    '3D-T2-FLAIR SAG NUEVO-1'

]
LUMBAR_PROTOCOLS_ACEPTED = {
    't2_tse_sag_384': 556,
    't2_tse_tra_384': 534,
    't1_tse_sag_320': 523,
    't1_tse_tra': 518,
}

options_dcm2niix = "-w 0 -i n -m y -ba n -f %x_%s -z y -g y"

def create_directory_mids_v1(xnat_data_path, mids_data_path, body_part, debug_level):
    # if debug_level == 3:
    #     pass
            # # Set up logging configuration
            # logging.basicConfig(filename='logfile_MIDS.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

            # # Create a logger
            # logger_MIDS = logging.getLogger("MIDS")

    with Path('logfile_MIDS.log').open("w") as file_:
        file_.write("")


    procedure_class_mr = ProceduresMR()
    procedure_class_light = LightProcedure()
    procedure_class_radiology = RadiologyProcedure()
    for subject_xnat_path in tqdm(xnat_data_path.iterdir()):
        print(f"{subject_xnat_path.name=}")
        procedure_class_mr.reset_indexes()
        procedure_class_light.reset_indexes()
        procedure_class_radiology.reset_indexes()
        for sessions_xnat_path in subject_xnat_path.iterdir():
            print(f"\t{sessions_xnat_path.name=}")
            findings = re.search(subses_pattern, str(sessions_xnat_path), re.X)
            subject_name = f"sub-{subject_xnat_path.stem}" 
            session_name = f"ses-{sessions_xnat_path.stem}" 

            mids_session_path = mids_data_path.joinpath(subject_name, session_name)
            xml_session_rois = list(sessions_xnat_path.rglob('*.xml'))
            
            tagger = Tagger()
            tagger.load_table_protocol(
                './xnat2mids/protocols/protocol_RM_prostate_train.tsv'
            )
            if not sessions_xnat_path.joinpath("scans").exists(): continue
            for scans_path in sessions_xnat_path.joinpath("scans").iterdir():
                    path_dicoms= list(scans_path.joinpath("resources").rglob("*.dcm"))[0].parent
                    dict_json = generate_json_dicom(path_dicoms)
                    if debug_level == 2: continue
                    modality = dict_json.get("Modality", "n/a")
                    series_description = dict_json.get("SeriesDescription", "n/a")
                    Protocol_name = dict_json.get("ProtocolName", "n/a")
                    image_type = dict_json.get("ImageType", "n/a")
                    body_part = dict_json.get("BodyPartExamined", body_part).lower()
                    acquisition_date_time = dict_json.get("AcquisitionDateTime", "n/a")
                    if acquisition_date_time == "n/a" or acquisition_date_time == "":
                        acquisition_date = dict_json.get("AcquisitionDate", "n/a")
                        acquisition_time = dict_json.get("AcquisitionTime", "n/a")
                        if acquisition_date == "n/a" or acquisition_date == "":
                            acquisition_date = "15000101"
                        if acquisition_time == "n/a" or acquisition_time == "":
                            acquisition_time = "000000"
                        acquisition_date_time = acquisition_date + acquisition_time
                    print(acquisition_date_time)
                    if "." not in acquisition_date_time:
                        acquisition_date_time += ".000000"
                    print(acquisition_date_time)  
                    if "T" in acquisition_date_time:
                        acquisition_date_time = datetime.strptime(acquisition_date_time, "%Y-%m-%dT%H:%M:%S.%f")
                    else:
                        acquisition_date_time = datetime.strptime(acquisition_date_time, "%Y%m%d%H%M%S.%f")
                    if modality == "MR":
                        #convert data to nifti
                        #if series_description not in BIOFACE_PROTOCOL_NAMES: continue
                        folder_conversion = dicom2nifti(path_dicoms)
                        # via BIDS protocols
                        searched_prost = prostate_pattern_comp.search(series_description)
                        if searched_prost and "tracew" not in series_description.lower():

                            print(f"series_description: {series_description}")
                            json_adquisitions = {
                                f'{k}': dict_json.get(k, -1) for k in dict_mr_keys.keys()
                            }
                            try:
                                protocol, acq, task, ce, rec, dir_, part, folder_BIDS = tagger.classification_by_min_max(json_adquisitions)
                                print(f"{protocol=}, {acq=}, {task=}, {ce=}, {rec=}, {dir_=}, {part=}, {folder_BIDS=}")
                                if protocol == "n/a":
                                    raise EmptyDataError("protocol is n/a")
                            except EmptyDataError as e:
                                print(f"EmptyDataError: {e}")
                                #print(f"{protocol=}, {acq=}, {task=}, {ce=}, {rec=}, {dir_=}, {part=}, {folder_BIDS=}")
                                continue
                            except KeyError as e:
                                print(f"KeyError: {e}")
                                continue
                            procedure_class_mr.control_sequences(
                                folder_conversion, mids_session_path, session_name, dict_json, protocol, acq, dir_, folder_BIDS, acquisition_date_time, body_part
                            )
                        
                            with Path('logfile_MIDS.log').open("a") as file_:
                                file_.write(f"{series_description}--->{(protocol, acq, task, ce, rec, dir_, part, folder_BIDS)}\n\t{path_dicoms}\n")
                            
                    if modality in ["OP", "SC", "XC", "OT", "SM", "BF"]: # opt , oct
                        
                        
                        
                        modality_, mim= (("op", "mim-light/op") if modality in ["OP", "SC", "XC", "OT"] else ("BF", "micr"))
                        if modality_ == "op":
                            folder_conversion = dicom2png(path_dicoms) #.joinpath("resources")
                        else:
                            folder_conversion = path_dicoms
                        laterality = dict_json.get("Laterality")
                        acq = "" if "ORIGINAL" in image_type else "opacitysubstract"
                        procedure_class_light.control_image(
                            folder_conversion, 
                            mids_session_path.joinpath(mim), 
                            dict_json, 
                            session_name, 
                            modality_,
                            acq, 
                            laterality, 
                            acquisition_date_time, 
                            body_part)
                    
                    if modality in ["CR", "DX"]:
                        try:
                            folder_conversion = dicom2png(path_dicoms) #.joinpath("resources")
                        except RuntimeError as e:
                            continue
                        modality_, mim= ((modality, f"mim-light/{modality.lower()}"))
                        laterality = dict_json.get("Laterality")
                        procedure_class_radiology.control_image(
                            folder_conversion, 
                            mids_session_path.joinpath(mim), 
                            dict_json, 
                            session_name, 
                            modality_,
                            laterality,
                            acquisition_date_time, 
                            body_part
                        )
                    
        if debug_level == 3: 
            continue
        print(f"copy sessions of {subject_name=}")
        procedure_class_mr.copy_sessions(subject_name)
        procedure_class_light.copy_sessions(subject_name)
        procedure_class_radiology.copy_sessions(subject_name)

participants_header = ['participant_id', 'participant_pseudo_id', 'modalities', 'body_parts', 'patient_birthday', 'age', 'gender']
participants_keys = ['PatientID','Modality', 'BodyPartExamined', 'PatientBirthDate', 'PatientSex', 'AcquisitionDateTime']
session_header = ['session_id','session_pseudo_id', 'acquisition_date_Time','radiology_report']
sessions_keys = ['AccessionNumber', 'AcquisitionDateTime']
scans_header = [
    'scan_file','BodyPart','SeriesNumber','AccessionNumber',
    'Manufacturer','ManufacturerModelName',
    'MagneticFieldStrength','ReceiveCoilName','PulseSequenceType',
    'ScanningSequence','SequenceVariant','ScanOptions','SequenceName','PulseSequenceDetails','MRAcquisitionType',
    'EchoTime','InversionTime','SliceTiming','SliceEncodingDirection','FlipAngle'
]
scans_header_micr = ['scan_file','BodyPart','SeriesNumber','AccessionNumber','Manufacturer','ManufacturerModelName','Modality', 'Columns','Rows','PhotometricInterpretation','ImagedVolumeHeight', 'ImagedVolumeHeight']
scans_header_op = ['scan_file','BodyPart','SeriesNumber','AccessionNumber','Manufacturer','ManufacturerModelName','Modality', 'Columns','Rows','PhotometricInterpretation', 'Laterality', 'note']

def create_tsvs(xnat_data_path, mids_data_path, body_part_aux):
    """
        This function allows the user to create a table in format ".tsv"
        whit a information of subject
        """
    
    list_information= []
    for subject_path in mids_data_path.glob('*/'):
        print(subject_path.name)
        if not subject_path.match("sub-*"): continue
        subject = subject_path.parts[-1]
        old_subject =subject.split("-")[-1]
        list_sessions_information = []
        modalities = []
        body_parts = []
        patient_birthday = None
        patient_ages = list([])
        patient_sex = None
        adquisition_date_time = None
        for session_path in subject_path.glob('*/'):
            if not session_path.match("ses-*"): continue
            session = session_path.parts[-1]
            
            old_sesion = session.split("-")[-1]
            
            
            report_path = list(xnat_data_path.glob(f'*{old_subject}/*{old_sesion}/**/sr_*.txt'))
            if not report_path:
                report="n/a"
            else:
                with report_path[0].open("r", encoding="iso-8859-1") as file_:
                    report = file_.read()
                report = report.replace("\t", "    ")
            list_scan_information = []
            for json_pathfile in session_path.glob('**/*.json'):
                note_path = json_pathfile.parent.joinpath(json_pathfile.stem + ".txt")
                note=""
                if note_path.exists():
                    with note_path.open('r') as file_:
                        note = file_.read()
                    note_path.unlink()
                    if not note:
                        print(f"empty note: {note_path}")
                        #raise FileNotFoundError
                else: 
                    print(f"not found: {note_path}")
                    #raise FileNotFoundError
                print(note)
                chunk_search = chunk_pattern_comp.search(json_pathfile.stem)
                if chunk_search:
                    list_nifties = json_pathfile.parent.glob(
                        chunk_pattern_comp.sub(
                            "*", 
                            json_pathfile.stem
                        ) + "*"
                    )
                else:
                    list_nifties = json_pathfile.parent.glob(
                        json_pathfile.stem + "*"
                    )
                
                list_nifties = [f for f in list_nifties if ".json" not in f.suffixes]
                print(json_pathfile)
                json_file = load_json(json_pathfile)
                print(json_file)
                pseudo_id = json_file[participants_keys[0]]
                modalities.append(json_file[participants_keys[1]])
                try:
                    body_parts.append(json_file[participants_keys[2]].lower())
                except KeyError as e:
                    body_parts.append(body_part_aux.lower())
                try:
                    patient_birthday = datetime.fromisoformat(json_file[participants_keys[3]])
                except KeyError as e:
                    patient_birthday = "n/a"
                except ValueError as e:
                    birtday = json_file[participants_keys[3]]
                    if birtday:
                        correct_birtday = f"{birtday[0:4]}-{birtday[4:6]}-{birtday[6:8]}"
                        patient_birthday = datetime.fromisoformat(correct_birtday)
                    else:
                        patient_birthday = "n/a"
                print(f"{patient_birthday=}, {patient_birthday.date()=}")
                try:
                    patient_sex = json_file[participants_keys[4]]
                except KeyError as e:
                    patient_sex = "n/a"
                acquisition_date_time = json_file.get("AcquisitionDateTime", "n/a")
                if acquisition_date_time == "n/a" or acquisition_date_time == "":
                    acquisition_date = json_file.get("AcquisitionDate", "n/a")
                    acquisition_time = json_file.get("AcquisitionTime", "n/a")
                    if acquisition_date == "n/a" or acquisition_date == "":
                        acquisition_date = "15000101"
                    if acquisition_time == "n/a" or acquisition_time == "":
                        acquisition_time = "000000"
                    acquisition_date_time = acquisition_date + acquisition_time
                if "." not in acquisition_date_time:
                    acquisition_date_time += ".000000"
                    
                if "T" in acquisition_date_time:
                    acquisition_date_time = datetime.strptime(acquisition_date_time, "%Y-%m-%dT%H:%M:%S.%f")
                else:
                    acquisition_date_time = datetime.strptime(acquisition_date_time, "%Y%m%d%H%M%S.%f")
                if patient_birthday != "n/a" and acquisition_date_time != "n/a":
                    patient_ages.append(int((acquisition_date_time - patient_birthday).days / (365.25)))

                if json_file[participants_keys[1]] == 'MR':
                    for nifti in list_nifties:
                        list_scan_information.append({
                            key:value
                            for key, value in zip(
                                scans_header,
                                [
                                    str(nifti.relative_to(session_path)),
                                    body_parts[-1], 
                                    *[json_file.get(key, "n/a") for key in scans_header[2:]],
                                    note
                                ]
                            )
                        })
                if json_file[participants_keys[1]] in ["OP", "SC", "XC", "OT"]:
                    for nifti in list_nifties:
                        list_scan_information.append({
                            key:value
                            for key, value in zip(
                                scans_header_op,
                                [
                                    str(nifti.relative_to(session_path)),
                                    body_parts[-1], 
                                    *[json_file.get(key, "n/a") for key in scans_header_op[2:-1]],
                                    note
                                ]
                            )
                        })
                if json_file[participants_keys[1]] in ["SM"]:
                    for nifti in sorted(list_nifties):
                        list_scan_information.append({
                            key:value
                            for key, value in zip(
                                scans_header_micr,
                                [
                                    str(nifti.relative_to(session_path)),
                                    body_parts[-1], 
                                    *[json_file.get(key_, "n/a") for key_ in scans_header_micr[2:]],
                                    note
                                ]
                            ) 
                        

                    })

            patient_ages = sorted(list(set(patient_ages)))
            modalities = sorted(list(set(modalities)))
            body_parts = sorted(list(set(body_parts)))
            try:
                accesion_number =  json_file['AccessionNumber']
            except KeyError:
                accesion_number = "n/a"
            print("acquisition_date_time", str(acquisition_date_time))
            list_sessions_information.append({
                 key:value
                 for key, value in zip(
                    session_header,
                    [session, accesion_number, acquisition_date_time.isoformat(), report]
                 )

            })
            pandas.DataFrame.from_dict(list_scan_information).to_csv(
                session_path.joinpath(f"{subject}_{session}_scans.tsv"), sep="\t", index=False
            )
        pandas.DataFrame.from_dict(list_sessions_information).to_csv(
            subject_path.joinpath(f"{subject}_sessions.tsv"), sep="\t", index=False
        )
        print(f"{patient_birthday=}")
        list_information.append({
            key:value
            for key, value in zip(
                participants_header,
                [subject, pseudo_id, modalities, body_parts, (str(patient_birthday.date()) if patient_birthday != "n/a" else patient_birthday), patient_ages, patient_sex]
            )
        })
    pandas.DataFrame.from_dict(list_information).to_csv(
        mids_data_path.joinpath("participants.tsv"), sep="\t", index=False
    )