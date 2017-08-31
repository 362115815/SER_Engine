#!/usr/bin/env python
# encoding: utf-8

import os



class ArffReader:
    def __init__(self,filelist):
        self.data=[]
        for file in filelist:
            with open(file,"r") as fin:
                temp_data=fin.readlines()
                index=-1
                flag=False 
                for item in temp_data:
                    index=index+1
                    if "@data" in item:
                        flag=True
                        break
            if(flag==False):
                raise Exception("无法找到 @data")
            temp_data=temp_data[index+2:]
            self.data.extend(temp_data)              

        '''
        with open(filepath,"r") as fin:
            self.data=fin.readlines()
            index=-1
            flag=False
            for item in self.data:
                index=index+1
                if "@data" in item:
                    flag=True
                    break
            if(flag==False):
                raise Exception("无法找到 @data")      
            self.data=self.data[index+2:]
        '''
    def add_data(self,filelist):
        for file in filelist:
            with open(file,"r") as fin:
                temp_data=fin.readlines()
                index=-1
                flag=False 
                for item in temp_data:
                    index=index+1
                    if "@data" in item:
                        flag=True
                        break
            if(flag==False):
                raise Exception("无法找到 @data")
            temp_data=temp_data[index+2:]
            self.data.extend(temp_data)                

