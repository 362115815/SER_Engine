import os
import shutil 
iemo_path="D:/xiaomin/IEMOCAP_full_release/IEMOCAP_full_release"

des_path="D:/xiaomin/data/IEMOCAP"

emo_class=[]

emo_count_by_person={}


emo_class_count={}



count=0

for i in range(1,6):
    session="Session"+str(i)
    session=os.path.join(iemo_path,session)
    emo_eval_path=session+"/dialog/EmoEvaluation"
    filelist=os.listdir(emo_eval_path)
    for filename in filelist:
        filepath=os.path.join(emo_eval_path,filename)
        if not os.path.isdir(filepath):
            print(filepath)
            with open(filepath,'r') as fin:
                data=fin.readlines()
                for item in data:
                    if "Ses" in item:
                        temp=item.strip().split('\t')
                        seg_name=temp[1]
                        emo_label=temp[2]
                        person=seg_name.split('_',1)[0]
                        if not person in emo_count_by_person.keys():
                            emo_count_by_person.setdefault(person,{})
                        if not emo_label in emo_count_by_person[person].keys():
                            emo_count_by_person[person].setdefault(emo_label,0)
                        if not emo_label in emo_class:
                            emo_class.append(emo_label)
                            os.mkdir(os.path.join(des_path,emo_label))
                            emo_class_count.setdefault(emo_label, 0)
                        emo_count_by_person[person][emo_label]=emo_count_by_person[person][emo_label]+1
                        emo_class_count[emo_label]=emo_class_count[emo_label]+1
                        temp=seg_name[:seg_name.rfind('_')];
                        wav_file=os.path.join(session,"sentences\wav",temp,seg_name+".wav")
                        shutil.copyfile(wav_file,os.path.join(des_path,emo_label,seg_name+"_"+emo_label+".wav"))
                        count=count+1
                        #print(wav_file)  
print(count)
with open(des_path+"/info.txt",'w') as fin:
    fin.write("total files :%d\n"%(count))
    for key,value in emo_class_count.items():
        fin.write("%s:%d\n"%(key,value))
    for key,item in emo_count_by_person.items():
        fin.write("\n"+key+":\n\n")
        for key1,value1 in item.items():
            fin.write("%s:%d\n"%(key1,value1))





    