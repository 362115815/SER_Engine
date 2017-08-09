#!/usr/bin/env python
# encoding: utf-8
import os

import matplotlib.pyplot as plt


out_dir="D:/test"
log_path="D:/2017-08-08_09_37_02.log"

plot_data={}
cv_num=0
cv_data={'train':[],'val':[],'cost':[]}

with open(log_path,'r') as fin:
    data=fin.readlines()
des_path=out_dir+'/'+os.path.splitext(os.path.basename(log_path))[0]

if not os.path.exists(des_path):
    os.mkdir(des_path)


flag=False
index=0
while index < len(data):
    item=data[index]
    if "Begin CV" in item:
        cv_num=cv_num+1
        if len(cv_data['train'])!=0:
            plot_data[cv_name]=dict(cv_data)
            cv_data['train']=[]
            cv_data['val']=[]
            cv_data['cost']=[]
        cv_name ="CV"+item.split(" ",3)[2]
    if "Strart Epoch" in item:
        for index_1 in range(2,5):
            #print(index_1)
            item=data[index+index_1]
            
            key=item.split(" ")[0].split('_')[-1]
            value=item.split(" ")[-1]
            cv_data[key].append(value)
        index=index+5
        continue
    index=index+1

plot_data[cv_name]=cv_data
print("cv_num:%d"%cv_num) 


for key,value in plot_data.items():
    plt.figure()
    plt.title(key)
    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0,len(value['train']))
    plt.plot(value['train'],label="train")
    plt.plot(value['val'],label="val")
    plt.legend(loc='upper left')
    plt.subplot(212)
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.xlim(0,len(value['train']))
    plt.plot(value['cost'])
    plt.suptitle(key)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.savefig(des_path+'/'+key+".png",dpi=300)
    print(len(value['train']))
    print(key)
