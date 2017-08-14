import os

in_dir =  "/data/mm0105.chen/wjhan/database/record_openInn_NOISE"                                                            
smilebin_path="/data/mm0105.chen/wjhan/xiaomin/opensmile/bin/SMILExtract"
config_path="/data/mm0105.chen/wjhan/xiaomin/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf"
output_dir="/data/mm0105.chen/wjhan/xiaomin/feature/intern_noise"


wav_list_path="/data/mm0105.chen/wjhan/database/record_openInn_NOISE.list"


with open(wav_list_path,'r') as fin:
    wav_list=fin.readlines()

scene_include=['office']



print("file_num:%d"%len(wav_list))

other_cmd=" -classes {ang,hap,nor,sad}"

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
    output_dir+"/intern_noise.arff"+" -instname "+instname +" -classlabel "+class_label+other_cmd
    print(cmd)
    os.system(cmd)
    count=count+1   
print("extracted count=%d"%(count))





