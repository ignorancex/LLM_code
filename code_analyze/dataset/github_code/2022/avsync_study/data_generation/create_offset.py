import os
import glob

video_files = glob.glob("./baseball*.mp4")

for input_file in video_files:
    base = os.path.basename(input_file)
    for offset in [0.1,0.5,1,-0.1,-0.5,-1]:
        output_file = os.path.join('./output_offset', os.path.splitext(base)[0] + '_offset_' + str(offset) + '.mp4')
        cmd = "ffmpeg -y -i '{input}' -itsoffset {offset} -i '{input}' -map 1:v -map 0:a -c copy '{output}'".format(input=input_file,offset=offset,output=output_file)
        print(cmd)
        os.system(cmd)
