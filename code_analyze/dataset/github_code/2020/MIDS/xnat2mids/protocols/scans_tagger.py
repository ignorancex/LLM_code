import pandas as pd
import numpy as np
import json
pd.set_option('display.max_columns', None)
from pandas.errors import EmptyDataError
class Tagger:
    # scan group
    #               SE = Spin Echo          =====>>>> 'SE'
    #               IR = Inversion Recovery =====>>>> 'IR' 'IR\\SE' 'SE\\IR'
    #               GR = Gradient Recalled  =====>>>> 'GR'
    #               EP = Echo Planar        =====>>>> 'EP' 'EP\\SE' 'EP\\SE\\EP' 'EP\\S'
    #               RM = Research Mode      =====>>>> 'RM'

    def __init__(self):

        self.ScanSeq = [['SE'], ['IR', 'IR\\SE', 'SE\\IR'], ['GR', 'GR\\IR', 'IR\\GR', 'EP\\GR'],
                        ['EP', 'EP\\SE', 'EP\\SE\\EP', 'EP\\S', 'EP\\G', 'EP\\IR'], ['RM']]

    def load_table_protocol(self, protocol_table_path):
        self.table_protocols = pd.read_csv(protocol_table_path, sep='\t', index_col=False)
        
    def classification_by_min_max(self, dict_atrubutes):
        print(dict_atrubutes)
        # targetScan = pd.DataFrame.from_dict([dict_atrubutes])
        # Verify to which scan group corresponds
        Manufacturer = dict_atrubutes["Manufacturer"]
        Manufacturers_model_name = dict_atrubutes["ManufacturerModelName"]
        scaning_sequence = dict_atrubutes["ScanningSequence"]
        sequence_variant = dict_atrubutes["SequenceVariant"]
        scan_options = dict_atrubutes["ScanOptions"]
        image_type = dict_atrubutes["ImageType"]
        series_description = dict_atrubutes["SeriesDescription"]
        if series_description == ".": raise EmptyDataError
        print(f"{json.dumps(Manufacturer)}")
        if Manufacturer == "Agfa": raise EmptyDataError
        
        print(f"{json.dumps(Manufacturers_model_name)}")
        print(f"{json.dumps(scaning_sequence)}")
        print(f"{json.dumps(sequence_variant)}")
        print(f"{json.dumps(scan_options)}")
        print(f'"{json.dumps(image_type)}"')
        #scaning_sequence = scaning_sequence if type(scaning_sequence) is str else "\\".join(scaning_sequence)
        

        table_protocol_M = self.table_protocols[[
            any([True for s in json.loads(l) if s == Manufacturer])
            for l in list(self.table_protocols["Manufacturer"])
        ]]
        # print("#" * 40, "table_protocol_M", "#" * 40)
        # print(table_protocol_M[["Protocol", "acq", "Manufacturer"]])

        table_protocol_M_MMN = table_protocol_M[[
            any([True for s in json.loads(l) if s == Manufacturers_model_name])
            for l in list(table_protocol_M["ManufacturerModelName"])
        ]]
        # print("#" * 40, "table_protocol_M_MMN", "#" * 40)
        # print(table_protocol_M_MMN[["Protocol", "acq", "Manufacturer","ManufacturerModelName"]])

        if table_protocol_M_MMN.empty: raise EmptyDataError 

        table_protocol_M_MMN_SS = table_protocol_M_MMN[[
            any([True for s in json.loads(l) if s == scaning_sequence])
            for l in list(table_protocol_M_MMN["ScanningSequence"])
        ]]

        # print("#" * 40, "table_protocol_M_MMN_SS", "#" * 40)
        # print(table_protocol_M_MMN_SS[["Protocol", "acq", "Manufacturer", "ManufacturerModelName"]])
        
        table_protocol_M_MMN_SS_SV = table_protocol_M_MMN_SS[[
            any([True for s in json.loads(l) if s == sequence_variant])
            for l in list(table_protocol_M_MMN_SS["SequenceVariant"])
        ]]
        
        # print("#" * 40, "table_protocol_M_MMN_SS_SV", "#" * 40)
        # print(table_protocol_M_MMN_SS_SV[["Protocol", "acq", "Manufacturer", "ManufacturerModelName"]])

        table_protocol_M_MMN_SS_SV_SO = table_protocol_M_MMN_SS_SV[[
            any([True for s in json.loads(l) if s == scan_options])
            for l in list(table_protocol_M_MMN_SS_SV["ScanOptions"])
        ]]


        # print("#" * 40, "table_protocol_M_MMN_SS_VS_SO", "#" * 40)
        # print(table_protocol_M_MMN_SS_SV_SO[["Protocol", "acq", "Manufacturer", "ManufacturerModelName"]])

        #for l in list(table_protocol_M_MMN_SS_SV_SO["ImageType"]):
            #print(l)
            #print(json.loads(l))
        
        table_protocol_M_MMN_SS_SV_SO_IT = table_protocol_M_MMN_SS_SV_SO[[
            any([True for s in json.loads(l) if s == image_type])
            for l in list(table_protocol_M_MMN_SS_SV_SO["ImageType"])
        ]]

        # print("#" * 40, "table_protocol_M_MMN_SS_VS_SO_IT", "#" * 40)
        # print(table_protocol_M_MMN_SS_SV_SO_IT[["Protocol", "acq", "Manufacturer"]])

        table_protocols = table_protocol_M_MMN_SS_SV_SO_IT
        # print("#"*40, "table_protocol_SS", "#"*40)
        # print(table_protocol_SS)
        if table_protocols.empty:
           return ["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]
        
        # print(f"{dict_atrubutes=}")
        matrix = []
        # print(list(dict_atrubutes.keys())[-5:])
        adquisition_param_keys = list(dict_atrubutes.keys())[-6:-2]
        for p in adquisition_param_keys:
            distance = []
            # print("p_value", dict_atrubutes[p], type(dict_atrubutes[p]))
            
            p_value = float(dict_atrubutes[p])  # if dict_atrubutes[p] !="nan" else -1
            for list_values in table_protocols[p]:
                min_ = np.amin(eval(list_values))
                max_ = np.amax(eval(list_values))
                if p_value >= min_ and p_value <= max_:
                    distance.append(np.sum([0., 0.]))
                else:
                    distance.append(np.sum([abs(p_value - min_), abs(p_value - max_)]))
            matrix.append(distance)
        
        pos_table_protocol = np.argmin(np.array(matrix).sum(axis=0))
        return table_protocols.iloc[pos_table_protocol][["Protocol", "acq", "task", "ce", "rec", "dir", "part", "folder"]].fillna('')

        [["DERIVED", "PRIMARY", "DIFFUSION", "CALC_BVALUE", "TRACEW", "NORM", "DIS2D", "DFC", "MIX", "MFSPLIT"]["DERIVED", "PRIMARY", "DIFFUSION", "CALC", "BVALUE", "TRACEW", "NORM", "DIS2D", "DFC", "MIX", "M"], ["DERIVED", "PRIMARY", "DIFFUSION", "CALC", "BVALUE", "TRACEW", "NORM", "DIS2D", "MFSPLIT"], ["DERIVED", "PRIMARY", "DIFFUSION", "NONE", "TRACEW", "DIS2D"], ["DERIVED", "PRIMARY", "DIFFUSION", "ADC", "NORM", "DIS2D", "MFSPLIT"], ["DERIVED", "PRIMARY", "DIFFUSION", "ADC", "DIS2D"], ["DERIVED", "PRIMARY", "DIFFUSION", "ADC", "NORM", "DIS2D", "DFC", "MIX", "MFSPLIT"]]