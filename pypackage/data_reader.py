#!/usr/bin/env python
# encoding: utf-8

import os



class ArffReader:
    def __init__(self,filepath):
        with open(filepath,"r") as fin:
            self.data=fin.readlines()
            index = self.data.index("@data\r\n")       
            self.data=self.data[index+2:]

