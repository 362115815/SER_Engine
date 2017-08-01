import os
in_dir="D:/rec2" #wav文件目录

out_dir="D:/wav_cut/rec2"

for _,_,filenames in os.walk(in_dir):
    for filename in filenames:
        cmd="SMILExtractPA -C D:/opensmile/config/myconfig/vad.conf  -I "+in_dir+'/'+filename+" -filebase "+out_dir+"/"+filename.replace(".wav","_")+" 2>nul"
        os.system(cmd)
