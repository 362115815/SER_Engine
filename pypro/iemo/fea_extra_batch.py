import os
in_dir = "D:/xiaomin/data/IEMOCAP/"
smilebin_path="D:/opensmile/bin/SMILExtractPA"
config_path="D:/opensmile/config/gemaps/eGeMAPSv01a.conf"
output_dir="D:/xiaomin/data"


other_cmd=" -l 0 -classes {ang,dis,exc,fea,fru,hap,neu,oth,sad,sur,xxx} "

count=0


for _,subdirs,filenames in os.walk(in_dir):
    for subdir in subdirs:
        dir_path=os.path.join(in_dir,subdir)
        filelist=os.listdir(dir_path)
        for filename in filelist:
            instname=filename[:filename.rfind(".")]
            in_file=os.path.join(dir_path,filename)
            cmd=smilebin_path+" -C "+config_path+" -I "+in_file+" -O "+\
            output_dir+"/iemo.arff"+" -instname "+instname+" -classlabel "+subdir+other_cmd
            os.system(cmd)
            count=count+1

print("count=%d"%(count))
'''
    for filename in filenames:
        filepath=in_dir+"/"+filename
        cmd=smilebin_path+" -C "+config_path+" -I "+filepath+" -instname " + instname +" -classlabel "+ classlabel[0]
        cmd=cmd+filepath
        os.system(cmd)
'''