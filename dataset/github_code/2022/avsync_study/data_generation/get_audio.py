import os
import glob

video_files = glob.glob("./*.mp4")

for input_file in video_files:
    base = os.path.basename(input_file)
    output_file = os.path.join('./full_audio', os.path.splitext(base)[0] + '_audio.aac')
    cmd = "ffmpeg -y -i '{input}' -vn -acodec copy '{output}'".format(input=input_file, output=output_file)
    print(cmd)
    os.system(cmd)
