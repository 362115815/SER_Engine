#!/usr/bin/env python
# encoding: utf-8

from numpy import zeros
import random
import numpy as np
rootdir = 'D:/xiaomin/feature/iemo/washedS8'
 
out_dir='D:/xiaomin/feature/iemo/washedS8'

# 读入特征数据
fin = open(rootdir+"/iemo.arff","r")
feadata = fin.readlines()
fin.close()

index = feadata.index("@data\n")

feadata = feadata[index+2:]

print(type(feadata))

#按情感随机分


#统计情感个数和类别

emo_class={}

sample_id={}
sample_index={}

for item in feadata:
    emo=item.strip().split(",")[-1]
    if emo not in emo_class.keys():
        emo_class[emo]=0
    emo_class[emo]=emo_class[emo]+1


set_num=10
fout=[]
for i in range(set_num):
    fout.append(open(out_dir+'/'+str(i)+".txt",'w'))


for key in emo_class.keys():
    x=list(range(emo_class[key]))
    np.random.shuffle(x)
    sample_id[key]=x
    sample_index[key]=0

sample_num=0

for item in feadata:
    emo=item.strip().split(",")[-1]
    file_index=sample_id[emo][sample_index[emo]]%set_num
    sample_index[emo]=sample_index[emo]+1
    sample_num=sample_num+1
    fout[file_index].write(item)

for  file in fout:
    file.close()

print("sample_num:%d"%sample_num)







