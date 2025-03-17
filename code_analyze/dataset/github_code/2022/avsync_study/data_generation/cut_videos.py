import os
import glob
import pandas as pd

video_files = glob.glob("./baseball*.mp4")
cut_csv = pd.read_csv('../AVSync_reference-Cuts.csv')
video_base_list = cut_csv['content'].tolist()
start = cut_csv['start'].tolist()
length = cut_csv['length'].tolist()


for input_file in video_files:
#    offset = os.path.splitext(os.path.basename(input_file))[0].split('_')[2]

#    if(offset in ['-0.1','0.1','-0.05','0.05']):
    base = os.path.splitext(os.path.basename(input_file))[0]
    content = base.split('_')[0]
    index = video_base_list.index(content)
    start_time = start[index]
    length_time = length[index]

    output_file = os.path.join('./output_cuts', base + '_cut_' + str(start_time) + '.mp4')
    cmd = "ffmpeg -y -i '{input}' -ss {start_time} -t {length_time} '{output}'".format(input=input_file, start_time=start_time,length_time=length_time,output=output_file)
    print(cmd)
    os.system(cmd)
