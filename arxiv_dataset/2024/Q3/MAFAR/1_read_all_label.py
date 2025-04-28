
import os
import numpy as np
import json
result_add = "predictions_10s"
model_list = os.listdir(result_add)




for model in model_list:
    prediected_result_list = os.listdir(os.path.join(result_add, model))
    model_add = os.path.join(result_add, model)
    blank_label = {}
    for file in prediected_result_list:

        data = np.load(os.path.join(model_add ,file), allow_pickle=True)
        label_blank = data.shape[0]
        blank_label[file[:-4]] = [[[],[]] for i in range(label_blank)]

    print(blank_label)
    break

def parse_event_data(line):

    parts = line.strip().replace("\t",' ').replace('   ',' ').replace("  ",' ').replace(" - ",' ')
    # print(parts)
    parts = parts.split(' ')
    filename = parts[0]

    start_time = float(parts[1])
    end_time = float(parts[2])
    event_type = ' '.join(parts[3:])

    # event_type = event_type.replace(";",',')
    # event_type = event_type.replace("— ",'')
    # event_type = event_type.replace("— ",'')
    # event_type = event_type.split(" 和")[0]
    # event_type = event_type.split(" (")[0]
    print(event_type)
    # event_type = event_type.split(",")[0]
    event_type = event_type.split('&')
    return {
        'filename': filename,
        'start_time': start_time,
        'end_time': end_time,
        'event_type': event_type,
    }

def find_audio_clip(start_time,end_time,time_step=10,window_length=10):
    if start_time < window_length:
        start_index = 0
    else:
        start_index = int((start_time-window_length)/ time_step)+1

    if end_time < window_length:
        end_index = 0
    else:
        end_index = int((end_time-window_length)/ time_step)+1

    return start_index,end_index



all_events_name = []
label_add = "label"
all_file_list = os.listdir(label_add)
all_file_list.sort()

for file in all_file_list:
    file_add = os.path.join(label_add, file)
    all_person_label_list= os.listdir(file_add)
    for person in all_person_label_list:
        print(os.path.join(file_add,person))
        with open(os.path.join(file_add,person), 'r',encoding="utf-8") as file:
            lines = file.readlines()

        event_data = [parse_event_data(line) for line in lines]
        for event in event_data:

            file_name = event["filename"][:-4]
            parts = file_name.split("_")
            file_name = '_'.join(parts[:-1])
            start_time = event["start_time"]
            end_time = event["end_time"]
            event_type_list = event["event_type"]
            for event_type in event_type_list:
                if event_type not in all_events_name:
                    all_events_name.append(event_type.strip())
                start_index,end_index = find_audio_clip(start_time,end_time)
                for i in range(start_index,end_index+1):
                    blank_label[file_name][i][0].append(event_type.strip())
                    blank_label[file_name][i][1].append(person)
with open("{}_label.json".format(result_add),"w") as file:
    json.dump(blank_label, file)

with open("all_label_list.json","w") as file:
    json.dump(all_events_name, file)
all_events_name.sort()
