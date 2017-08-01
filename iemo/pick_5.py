#!/usr/bin/env python
# encoding: utf-8



from numpy import zeros
import random

rootdir = 'D:/xiaomin/feature/iemo'

emo_classes_5=['ang','sad','hap','exc','neu','fru']

emo_classes_4=['ang','sad','hap','exc','neu']


# 读入特征数据
fin = open(rootdir+"/iemo.arff","r")
feadata = fin.readlines()
fin.close()
emo_str_5="{ang,fru,hap,neu,sad}"
emo_str_4="{ang,hap,neu,sad}"

flag=True
with open(rootdir+"/iemo_4emo.arff","w") as fout:
    for item in feadata:
        if(flag):
            if "@attribute emotion" in item:
                item=item.replace("{ang,dis,exc,fea,fru,hap,neu,oth,sad,sur,xxx}",emo_str_4)
            fout.write(item)

            if "@data\n" in item :
                fout.write('\n')
                flag=False
               
        else:
            temp=item.strip().split(",")[-1]
            if temp in emo_classes_4:
                item=item.replace("exc","hap")
                fout.write(item)



