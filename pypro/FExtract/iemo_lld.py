import os

smilebin_path='SMILExtract'
config_path="/data/mm0105.chen/wjhan/xiaomin/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf"
output_dir="/data/mm0105.chen/wjhan/xiaomin/feature/iemo/lld"
filelist_path="/data/mm0105.chen/wjhan/database/srcb_wav_data/IEMO_denoise/msc/file.list"
input_dir='/data/mm0105.chen/wjhan/database/srcb_wav_data/IEMO_denoise/msc'

emo_classes = {'ang': 0, 'hap': 1, 'nor': 2, 'sad': 3,'neu':2,'exc':1}


with open(filelist_path,'r') as fin:
	filelist=fin.readlines()
count=0
for file in filelist:
	if file in ['.','..']:
		continue
	file=file.strip()
	label=file.split("_")[-2]
	in_file=os.path.join(input_dir,file)
	out_file=os.path.join(output_dir,file[:-5]+".csv")
	#print(out_file)
	print(label)
	cmd=smilebin_path+" -l 1 -I "+in_file+" -C "+config_path+" -lldcsvoutput "+ out_file +" -instname "+str(emo_classes[label])+" -headercsvlld 0"
	print(cmd)
	os.system(cmd)
	#exit()
	count=count+1
print("count:%d"%count)