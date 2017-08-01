import os

path_prefix = "D:/xiaomin/data/manAffSpch/"
fea_prefix = "D:/xiaomin/feature/"
config_path = "D:/xiaomin/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf"
opensmile_path="D:/xiaomin/opensmile-2.3.0/bin/Win32/"


fin = open("D:/xiaomin/data/manAffSpch/wavlist.txt", 'r')
wavlist = fin.readlines()
print(type(wavlist))

for filepath in wavlist :
    filepath=filepath.replace("\\","/")[:-1]

    filename = path_prefix+filepath

    instname =  filepath[filepath.find("/")+1:filepath.rfind(".")].replace("/", "_")

    classlabel = instname.split("_", 2)[1:2]

    feaname = fea_prefix + instname + ".arff"

    outputname=fea_prefix+"output.arff"

    command = opensmile_path + "SMILExtract" + " -C " + config_path + " -I " + filename + " -O " + \
              outputname + " -instname " + instname  + " -classes "  + "{anger,elation,neutral,panic,sadness}"+ " -classlabel "+ classlabel[0]+" -l 0"
    os.system(command)
