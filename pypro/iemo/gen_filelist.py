import os
in_dir = "D:/xiaomin/data/IEMOCAP/washed"

output_dir="D:/xiaomin/data/IEMOCAP/washed"
count=0
fout=open(output_dir+'/filelist.txt','w')

for _,subdirs,filenames in os.walk(in_dir):
    for subdir in subdirs:
        dir_path=os.path.join(in_dir,subdir)
        filelist=os.listdir(dir_path)
        for filename in filelist:
            in_file=os.path.join(dir_path,filename)
            fout.write(in_file+'\n')
            count=count+1
fout.close()
print("count=%d"%(count))
'''
    for filename in filenames:
        filepath=in_dir+"/"+filename
        cmd=smilebin_path+" -C "+config_path+" -I "+filepath+" -instname " + instname +" -classlabel "+ classlabel[0]
        cmd=cmd+filepath
        os.system(cmd)
'''