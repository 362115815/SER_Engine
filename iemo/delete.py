#!/usr/bin/python  
#coding:utf8  
  
import os  
  

def dirlist(path,count_d,count):  
    filelist =  os.listdir(path)  
  
    for filename in filelist:  
        filepath = os.path.join(path, filename)  
        if os.path.isdir(filepath):
           # print("dirpath=%s"%(filepath))  
            dirlist(filepath,count_d,count)  
        else:   
            if filename[0]=='.':
                os.remove(filepath)
                count_d=count_d+1
            else:
                count=count+1
                #exit()
    return count_d,count
if __name__=='__main__':
    path="D:/xiaomin/IEMOCAP_full_release/IEMOCAP_full_release"
    count_d,count=dirlist(path,0,0)
    print("file deleted:%d,file remain:%d"%(count_d,count))
