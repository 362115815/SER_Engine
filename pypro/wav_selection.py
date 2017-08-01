#!/usr/bin/env python
# encoding: utf-8
import os
workdir = 'D:/xiaomin/wav_selection'
corpuspath='D:/xiaomin/data/manAffSpch/data'
for i in range(10):
    if i==5 or i==8:
        continue
    filepath = 'D:/xiaomin/feature' + '/'+str(i)+'.txt'

    fin = open(filepath, 'r')

    data = fin.readlines()

    for item in data:
        id = item.split(',')[0]
        if 'utterance' not in id:
            continue
        id = id.strip('\'')

        wavpath= corpuspath+'/'+id.replace('_', '/')+".wav"

        label = id.split('_')[1]

        despath = workdir+"/train/"+label+'/' +id+'.wav'

        os.system("copy %s %s 1>nul " %(wavpath.replace('/', '\\'), despath.replace('/', '\\')))



