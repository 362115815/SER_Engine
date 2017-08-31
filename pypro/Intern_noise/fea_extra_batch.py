import os

in_dir = ""
smilebin_path="D:/opensmile/bin/SMILExtractPA"
config_path="D:/opensmile/config/gemaps/eGeMAPSv01a.conf"
output_dir="D:/xiaomin/feature/intern"

wav_list_path="D:/record_openInn_NOISE.list"

with open(wav_list_path,'r') as fin:
    wav_list=fin.readlines()

scene_include=['office']



print("file_num:%d"%len(wav_list))

other_cmd=" -l 0 -classes {ang,dis,exc,fea,fru,hap,neu,oth,sad,sur,xxx} "

count=0


for filename in wav_list:
    filename=filename.strip()
    flag=False
    for scene in scene_include:
        if scene in filename:
            flag=True
            break
    if flag==False:
        continue
    instname=filename[:filename.rfind(".")]
    in_file=os.path.join(in_dir,filename)
    class_label=instname.split("_",4)[-2][:3]
    cmd=smilebin_path+" -C "+config_path+" -I "+in_file+" -O "+\
    output_dir+"/intern_noise.arff"+" -instname "+instname+" -classlabel "+class_label+other_cmd
    os.system(cmd)
    count=count+1   

print("extracted count=%d"%(count))





