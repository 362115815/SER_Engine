import os
in_dir = "/data/mm0105.chen/wjhan/database/record_openInn_NOISE"
smilebin_path="/data/mm0105.chen/wjhan/xiaomin/opensmile-2.3.0/bin/SMILExtract1"
config_path="/data/mm0105.chen/wjhan/xiaomin/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf"
output_dir="/data/mm0105.chen/wjhan/xiaomin/feature/intern/withnoise"


other_cmd=" -l 0 -classes {ang,hap,nor,sad} "

count=0


for _,subdirs,filenames in os.walk(in_dir):

    for filename in filenames:
        instname=os.path.splitext(filename)[0]
        in_file=os.path.join(in_dir,filename)
        class_label=filename.split("_")[3][:3]
        

        cmd=smilebin_path+" -C "+config_path+" -I "+in_file+" -O "+\
        output_dir+"/intern.arff"+" -instname "+instname+" -classlabel "+class_label+other_cmd
        print(cmd)
        os.system(cmd)
        count=count+1
        exit()

print("count=%d"%(count))
'''
    for filename in filenames:
        filepath=in_dir+"/"+filename
        cmd=smilebin_path+" -C "+config_path+" -I "+filepath+" -instname " + instname +" -classlabel "+ classlabel[0]
        cmd=cmd+filepath
        os.system(cmd)
'''
