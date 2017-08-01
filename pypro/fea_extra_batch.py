import os
in_dir="D:/wav_cut/rec1"
smilebin_path="D:/opensmile/bin/SMILExtractPA"
config_path="D:/opensmile/config/gemaps/eGeMAPSv01a.conf"
output_dir=""

other_cmd=" -l 0 -classes {anger,elation,neutral,panic,sadness} -appendcsv 0"




for _,_,filenames in os.walk(in_dir):
    for filename in filenames:
        filepath=in_dir+"/"+filename

        cmd=smilebin_path+" -C "+config_path+" -I "+filepath+" -instname " + instname +" -classlabel "+ classlabel[0]

        cmd=cmd+filepath
        os.system(cmd)
