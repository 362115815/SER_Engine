import os
import shutil 
iemo_path="D:/xiaomin/data/IEMOCAP/washed/temp"

des_path="D:/xiaomin/data/IEMOCAP/washed"


emo_class_count={}

count=0

filelist=os.listdir(iemo_path)

for filename in filelist:
    if not os.path.isdir(filename):
        emo=filename.split(".")[0].split("_")[-1]
        print(emo)
        if emo not in emo_class_count.keys():
            emo_class_count[emo]=0
            os.mkdir(os.path.join(des_path,emo))
        count=count+1
        shutil.copyfile(os.path.join(iemo_path,filename),os.path.join(des_path,emo,filename))
        emo_class_count[emo]=emo_class_count[emo]+1


with open(des_path+"/info.txt",'w') as fin:
    fin.write("total files :%d\n"%(count))
    for key,value in emo_class_count.items():
        fin.write("%s:%d\n"%(key,value))


    