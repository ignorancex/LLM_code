import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sensor_bias_path = os.environ['SENSOR_BIAS_PATH']
#dataset_path = "/media/joshua/data/sensor_bias/images/"
dataset_path = f'{sensor_bias_path}/data_v3'
spreadsheet_path = f'{sensor_bias_path}/annotations/sensor_bias_data_v3.xlsx'

image_list = list(Path(dataset_path).rglob("*.jpg"))

shutter_values = ['1/4000', '1/2000', '1/1000', '1/500', '1/250', '1/125', '1/60', '1/30', '1/15', '1/8', '1/4', '0.5', '1', '2', '4', '8', '15', '30']
iso_values = [100, 200, 400, 800, 1600, 3200, 6400]
f_number_values = [22, 16, 11, 8, 5.6]

df = []

for image_path in image_list:
	image_path = os.path.relpath(str(image_path), dataset_path)
	print(image_path)

	dir_name, lux, ev_offset, fname = image_path.split("/")
	shutter_speed_idx, iso_idx, f_number_idx = fname.split(".jpg")[0].split("_")

	object_id, category = dir_name.split('_')

	record = {	'Path': image_path,
				'Object_name': dir_name,
				'Object_id': object_id,
				'Category': category,
				'EV_offset': int(ev_offset),
				'Lux': float(lux),
				'Shutter_speed': shutter_values[int(shutter_speed_idx)],
				'ISO': iso_values[int(iso_idx)],
				'F-Number': f_number_values[int(f_number_idx)]}
	df.append(record)

		#csv_entry = f"{image_path},{class_name},{date_label},{ev_offset},{shutter_speed},{iso},{f_number}\n"
		# print(csv_entry)
		#csv_file.write(csv_entry)
df = pd.DataFrame.from_dict(df)
df = df.sort_values(by=['Object_id', 'EV_offset', 'Lux'], ignore_index=True)
df.to_excel(spreadsheet_path, index=False)