#!/usr/bin/env python
# encoding: utf-8

from numpy import zeros
import random

rootdir = 'D:/xiaomin'

# 读入性别信息
fin = open(rootdir+"/feature/speakerInfo.txt", "r")
speakerInfo = fin.readlines()
fin.close()
genderInfo = {}
for item in speakerInfo:
    item = item.rstrip("\n")
    name, gender = item.split("\t")
    genderInfo[name] = gender


# 读入特征数据
fin = open(rootdir+"/feature/manAffSpch.arff","r")
feadata = fin.readlines()
fin.close()

index = feadata.index("@data\n")

feadata = feadata[index+2:]

print(type(feadata))

# 分成十份

F = (3, 3, 3, 2, 2, 2, 2, 2, 2, 2)
M = (4, 4, 4, 4, 4, 5, 5, 5, 5, 5)
num = len(M)
random.shuffle(speakerInfo)

divide = {}
cur_F = zeros(num, int)
cur_M = zeros(num, int)

d_list = {'Female': F, 'Male': M}

cur_list = {'Female': cur_F, 'Male': cur_M}


# 打开输出文件流
fout = []
for i in range(0, 10):
    fout.append(open(rootdir+"/feature/"+str(i)+".txt",'w'))

for item in feadata:
    name = item.lstrip("\'").split("_",1)[0]
    if name not in divide:
        while 1:
            index = random.randint(0, 9)
            gender = genderInfo[name]
            if cur_list[gender][index] < d_list[gender][index]:
                divide[name] = index
                cur_list[gender][index] = cur_list[gender][index] + 1
                break
    fout[divide[name]].write(item)


for i in range(0, 10):
    fout[i].close()






