#!/usr/bin/env python
# encoding: utf-8



from numpy import zeros
import random

rootdir = 'D:/xiaomin/feature/iemo'



# 读入特征数据
fin = open(rootdir+"/iemo.arff","r")
feadata = fin.readlines()
fin.close()

index = feadata.index("@data\n")

feadata = feadata[index+2:]

print(type(feadata))

#按说话者分

person=[]

out_file={}
count=0;
sample_count=0
fout=open(rootdir+"/info.txt","w")
for item in feadata:
    sample_count=sample_count+1
    name=item.split(",",1)[0]
    name=name.strip("\'").split("_",1)[0]
    if name not in person:
        person.append(name)
        fout.write("%d %s\n"%(count,name))
        out_file.setdefault(name,open(rootdir+"/"+str(count)+".txt","w"))
        count=count+1
    out_file[name].write(item)

for key in out_file.keys():
    out_file[key].close()
fout.close()
print("sample_count=%d"%(sample_count))








