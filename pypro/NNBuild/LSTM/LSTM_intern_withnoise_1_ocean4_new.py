#!/usr/bin/env python
# encoding: utf-8

import numpy as np 
import random 
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import sys
import datetime
import data_reader as dr
from test import mix_matrix

#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
'''
This is an LSTM network building script
'''



class CDataSet():
    def __init__(self, data, labels, shuffle=True):
        if len(data) == 0 or len(data) != len(labels):
            raise ValueError('data为空或data与label长度不匹配')
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self.is_shuffle = shuffle

    def _shuffle(self):
        c = list(zip(self.data, self.labels))
        random.shuffle(c)
        self.data, self.labels = zip(*c)
        print("shuffle finished")

    def next_batch(self, batch_size):  # 如果到达末尾，则把batch_size返回0，否则返回所读取的batch_size
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            return [], [], 0
        if (self.batch_id == 0):
            print("self.batch_id:", self.batch_id)
            if self.is_shuffle == True:
                self._shuffle()
        end_id = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_id])
        batch_labels = (self.labels[self.batch_id:end_id])
        num=end_id-self.batch_id
        self.batch_id = end_id
        return batch_data, batch_labels,num

# setting path
rootdir = '/data/mm0105.chen/wjhan/dzy/LSTM'
feadir = rootdir + '/feature'
logdir = rootdir + '/log'
modeldir = rootdir + '/model'

extra_train_set_path=["/data/mm0105.chen/wjhan/dzy/LSTM/feature/IEMObyxiomin/denoise/IEMO.arff"]
extra_val_set_path=''#["/data/mm0105.chen/wjhan/dzy/LSTM/feature/CHAEVD2.0/test/0.txt"]      #"/data/mm0105.chen/wjhan/xiaomin/feature/intern_noise/intern_noise_all.arff"

#     SRCB(29) + ENUS_0930  + ENUS_1010 + KOKR_1010 + speed(byperson_denoise_speed + ENUS_denoise_0930_speed + ENUS_denoise_1010_speed + KOKR_denoise_1010_speed))
train_set_path = ["/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/0930/ENUS_denoise_0930/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1010/denoies_1010/ENUS_denoise_1010/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1010/denoies_1010/KOKR_denoise_1010/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/intern/byperson_denoise_speed/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/0930/ENUS_0930_denoise_speed/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1010/ENUS_1010_denoise_speed/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1010/KOKR_1010_denoise_speed/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1013/ENUS_denoise_1013/",
                  #"/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1013/KOKR_denoise_1013/",
                  "/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1013/ENUS_1013_denoise_speed/",
                  #"/data/mm0105.chen/wjhan/dzy/LSTM/feature/ocean/1013/KOKR_1013_denoise_speed/"
                  ]
filepath_label = ['ENUS_denoise_0930','ENUS_denoise_1010', 'KOKR_denoise_1010', 'byperson_denoise_speed', 'ENUS_0930_denoise_speed', 'ENUS_1010_denoise_speed', 'KOKR_1010_denoise_speed',
                    'ENUS_denoise_1013','KOKR_denoise_1013','ENUS_1013_denoise_speed','KOKR_1013_denoise_speed']

#config
output_log=0   # 0 : 不打印log信息进log,只在桌面显示
save_model=0 # 0 ： 不保存模型
timestep_size = 1
corpus ='intern'
which_copy='byperson_denoise/feature'

do_dropout=1   # 增加 1层dropout层  , 以下面的概率随机丢弃一些输入数据，防止过拟合
_keep_prob=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# noise config
gender_include = ['M','F']
person_exclude=['03','07','08','09','15','18']
scene_include=['office']
scenario_include=['meet','white','seat']
db_include=['25']

# 每个隐含层的节点数
# 每个隐含层的节点数
hidden_size = [1024,512]     # 一层隐藏层：节点数为1024
acc_train_epsilon= 0.98
epoch_num =128           # 训练128轮
_batch_size=1024
learning_rate = 0.0005   #

# predefine
set_num = 29
ENUS_0930_denoise_nums = 17
ENUS_1010_denoise_nums = 11
KOKR_1010_denoise_nums = 15
ENUS_1013_denoise_nums = 12
KOKR_1013_denoise_nums = 15

byperson_denoise_speed_nums = 29
ENUS_0930_denoise_speed_nums = 17
ENUS_1010_denoise_spped_nums = 11
KOKR_1010_denoise_speed_nums = 15
ENUS_1013_denoise_speed_nums = 12
KOKR_1013_denoise_speed_nums = 15



fea_dim = 88		 # 88 维向量
emo_classes = {'ang': 0, 'hap': 1, 'nor': 2, 'sad': 3, 'neu':2, 'exc':1}

class_num =4         # 4类情感

now = datetime.datetime.now()

if output_log == 1:
    # 标准输入输出重定向到文本
    log_path=logdir + '/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.log'
    print("redirect output to %s"%log_path)
    savedStdout = sys.stdout
    fin = open(log_path, 'w+')
    sys.stdout = fin

print('*********************************************')
print('******* Run LSTM %s ********' % (now.strftime('%Y-%m-%d %H:%M:%S')))
print('*********************************************')

#print("SRCB + ENUS0930 + 1010(ENUS+KOKR) + 1013(ENUS+KOKR) + Speed(SRCB+ENUS0930+1010+1013) + IEMO,  shuffle all data")
print("SRCB + ENUS0930 + 1010(ENUS)+1013(ENUS)+ Speed(SRCB+ENUS0930+ENUS1010+ENUS1013) + IEMO,  shuffle all data")


# read data
data_set = []
data_set_speed = []
data_set_ENUS_denoise_0930 = []
data_set_ENUS_denoise_1010 = []
data_set_KOKR_denoise_1010 = []
data_set_ENUS_denoise_1013 = []
data_set_KOKR_denoise_1013 = []


# 训练集顺序： origin(29) + ENUS(0930) + ENUS(1010) + KOKR(1010)
# 读入训练集
for i in range(set_num):
    filepath = feadir+'/'+corpus+ '/' +which_copy+'/'+ str(i) + '.txt'
    with open(filepath, 'r') as fin:
        data_set.append(fin.readlines())
print("size of SRCB origin data_set:",  len(data_set))

# 读入数据变换后的训练集   ()
if len(train_set_path) != 0:
    for trainfiles_path in train_set_path:
        if filepath_label[0] in trainfiles_path:          # ENUS_0930_denoise
            for i in range(ENUS_0930_denoise_nums):
                ENUS_denoise_filepath_0930 = trainfiles_path + str(i) + '.txt'
                with open(ENUS_denoise_filepath_0930, 'r') as f_extra:
                    data_set_ENUS_denoise_0930.append(f_extra.readlines())
        if filepath_label[1] in trainfiles_path:          # ENUS_1010_denoise
            for i in range(ENUS_1010_denoise_nums):
                ENUS_denoise_filepath_1010 = trainfiles_path + str(i) + '.txt'
                with open(ENUS_denoise_filepath_1010, 'r') as f_extra:
                    data_set_ENUS_denoise_1010.append(f_extra.readlines())
        if filepath_label[2] in trainfiles_path:          # KOKR_1010_denoise
            for i in range(KOKR_1010_denoise_nums):
                KOKR_denoise_filepath_1010 = trainfiles_path + str(i) + '.txt'
                with open(KOKR_denoise_filepath_1010, 'r') as f_extra:
                    data_set_KOKR_denoise_1010.append(f_extra.readlines())
        if filepath_label[3] in trainfiles_path:          # byperson_denoise_speed
            for i in range(byperson_denoise_speed_nums):
                byperson_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(byperson_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
        if filepath_label[4] in trainfiles_path:   # ENUS_0930_denoise_speed
            for i in range(ENUS_0930_denoise_speed_nums):
                ENUS_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(ENUS_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
        if filepath_label[5] in trainfiles_path:    # ENUS_1010_denoise_speed
            for i in range(ENUS_1010_denoise_spped_nums):
                ENUS_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(ENUS_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
        '''
        if filepath_label[6] in trainfiles_path:    # KOKR_1010_denoise_speed
            for i in range(KOKR_1010_denoise_speed_nums):
                KOKR_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(KOKR_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
        '''
        if filepath_label[7] in trainfiles_path:    # ENUS_1013_denoise
            for i in range(ENUS_1013_denoise_nums):
                ENUS_1013_denoise_filespath = trainfiles_path + str(i) + '.txt'
                with open(ENUS_1013_denoise_filespath, 'r') as f_extra:
                    data_set_ENUS_denoise_1013.append(f_extra.readlines())
        if filepath_label[8] in trainfiles_path:    # ENUS_1013_denoise
            for i in range(KOKR_1013_denoise_nums):
                KOKR_1013_denoise_filespath = trainfiles_path + str(i) + '.txt'
                with open(KOKR_1013_denoise_filespath, 'r') as f_extra:
                    data_set_KOKR_denoise_1013.append(f_extra.readlines())
        if filepath_label[9] in trainfiles_path:  # ENUS_1013_denoise_speed
            for i in range(ENUS_1013_denoise_speed_nums):
                ENUS_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(ENUS_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
        if filepath_label[10] in trainfiles_path:  # KOKR_1010_denoise_speed
            for i in range(KOKR_1013_denoise_speed_nums):
                KOKR_denoise_speed_filepath = trainfiles_path + str(i) + '.txt'
                with open(KOKR_denoise_speed_filepath, 'r') as f_extra:
                    data_set_speed.append(f_extra.readlines())
print("size of data_set_ENUS_denoise_0930:", len(data_set_ENUS_denoise_0930))
print("size of data_set_ENUS_denoise_1010:", len(data_set_ENUS_denoise_1010))
print("size of data_set_KOKR_denoise_1010:", len(data_set_KOKR_denoise_1010))
print("size of data_set_ENUS_denoise_1013:", len(data_set_ENUS_denoise_1013))
print("size of data_set_KOKR_denoise_1013:", len(data_set_KOKR_denoise_1013))
print("szie of data_set_speed:(first byperson_denoise_speed, then ENUS_denoise_speed)", len(data_set_speed))



#读入额外训练集
if len(extra_train_set_path)!=0 :
    for set_path in extra_train_set_path:
        extra_train_set=dr.ArffReader(extra_train_set_path)

#读入额外验证集
if extra_val_set_path!="" :
    extra_val_set=dr.ArffReader(extra_val_set_path)


# CV
cv_vali_acc = []   # 每次CV 时val_set的准确率
cv_train_acc = [] # 每次CV 时train_set的准确率
cv_vali_acc_fouremo_list = []
cv_train_acc_fouremo_list = []

for i in range(set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_nums+KOKR_1010_denoise_nums): #29+17+11
    print('Begin CV %d :' % (i))    # 有多少个人就有多少个CV，每次CV时拿出一个人的数据来作为验证集，其余人的数据作为训练集，每个CV训练时最多训练128次，除非那次训练的
    if save_model==1:
        cv_dir = modeldir + '/' + now.strftime('%Y-%m-%d_%H_%M_%S') + '_cv' + str(i)
        os.system('mkdir ' + cv_dir)

    '''data prepare'''
    # create val_set,val_label,train_set,train_label
    val_set = []
    val_labels = []
    train_set = []
    train_labels = []

    # Read origin inner SRCB data (CV 0-28)
    for inner_txt_number in range(set_num): # 29
        for item in data_set[inner_txt_number]:
            temp = item.strip().split(',')
            name = temp[0].strip('\'').split('_')
            person = name[0][:2]
            gender = name[0][2]
            if person in person_exclude:
                continue
            if gender not in  gender_include:
                continue
            fea = [float(index) for index in temp[1:-1]]
            emo_label = temp[0].strip("\'").split("_")[-1]
            if emo_label not in emo_classes.keys():
                continue
            onehot_label = np.zeros(class_num)
            onehot_label[emo_classes[emo_label]] = 1
            if inner_txt_number == i:
                val_set.append(fea)
                val_labels.append(onehot_label)
            else:
                train_set.append(fea)
                train_labels.append(onehot_label)
    if len(val_set) == 0 and i < 29:
        print('inner val_set size = 0, skip CV%d' %i)
        continue
    print("after inner SRCB, train_set, val_set:",len(train_set), len(val_set))

    # read ENUS(0930) train data  （CV 29-45）
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if filepath_label[0] in trainfiles_path:
                for ENUS_0930_txt_number in range(ENUS_0930_denoise_nums):  # 17 (ENUS_0930_txt_number: 0-16)
                    for item in data_set_ENUS_denoise_0930[ENUS_0930_txt_number]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        gender = name[1]
                        if gender not in gender_include:
                            continue
                        fea = [float(index) for index in temp[1:-1]]
                        emo_label = temp[0].strip("\'").split("_")[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if (ENUS_0930_txt_number+inner_txt_number+1) == i:   # (0 + 28  + 1 )   ==  (29)i
                            val_set.append(fea)
                            val_labels.append(onehot_label)
                        else:
                            train_set.append(fea)
                            train_labels.append(onehot_label)
                if len(val_set) == 0 and i < 46:
                    print("ENUS val_set size = 0, skip CV%d" %i)
                    continue
                print("after add ENUS_0930, train_set, val_set:", len(train_set), len(val_set))

    # read ENUS _1010
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if filepath_label[1] in trainfiles_path:
                    for ENUS_1010_txt_number in range(ENUS_1010_denoise_nums):   # 11 (ENUS_1010_txt_number: 0-10)
                        for item in data_set_ENUS_denoise_1010[ENUS_1010_txt_number]:
                            temp = item.strip().split(',')
                            name = temp[0].strip('\'').split('_')
                            gender = name[1]
                            if gender not in gender_include:
                                continue
                            fea = [float(index) for index in temp[1:-1]]
                            emo_label = temp[0].strip("\'").split("_")[-1]
                            if emo_label not in emo_classes.keys():
                                continue
                            onehot_label = np.zeros(class_num)
                            onehot_label[emo_classes[emo_label]] = 1
                            if (ENUS_1010_txt_number+ENUS_0930_txt_number+inner_txt_number+2) == i:   # (0 +16 + 28 +2)
                                val_set.append(fea)
                                val_labels.append(onehot_label)
                            else:
                                train_set.append(fea)
                                train_labels.append(onehot_label)
                    if len(val_set) == 0 and i < 57:
                        print("ENUS_1010 val_set size = 0, skip CV%d" % i)
                        continue
                    print("after add ENUS_1010, train_set, val_set:", len(train_set), len(val_set))

    # read KOKR(1010) train data （CV 57 - 71）
    '''
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if filepath_label[2] in trainfiles_path:
                for KOKR_1010_txt_number in range(KOKR_1010_denoise_nums):  # 15 (KOKR_1010_denoise_nums: 0-14)
                    for item in data_set_KOKR_denoise_1010[KOKR_1010_txt_number]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        gender = name[1]
                        if gender not in gender_include:
                            continue
                        fea = [float(index) for index in temp[1:-1]]
                        emo_label = temp[0].strip("\'").split("_")[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if (KOKR_1010_txt_number+ENUS_1010_txt_number+ENUS_0930_txt_number+inner_txt_number+3) == i:   # (0 + 10 +16 + 28 +3) 57
                            val_set.append(fea)
                            val_labels.append(onehot_label)
                        else:
                            train_set.append(fea)
                            train_labels.append(onehot_label)
                if len(val_set) == 0 and i < 72:
                     print("KOKR_1010 val_set size = 0, skip CV%d" % i)
                     continue
                print("after add KOKR_1010, train_set, val_set:", len(train_set), len(val_set))
    '''

    # read ENUS(1013) train data （CV 72 - 83）
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if filepath_label[7] in trainfiles_path:
                for ENUS_1013_txt_number in range(ENUS_1013_denoise_nums):  # 12 (KOKR_1010_denoise_nums: 0-11)
                    for item in data_set_ENUS_denoise_1013[ENUS_1013_txt_number]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        gender = name[1]
                        if gender not in gender_include:
                            continue
                        fea = [float(index) for index in temp[1:-1]]
                        emo_label = temp[0].strip("\'").split("_")[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if (ENUS_1013_txt_number + KOKR_1010_txt_number + ENUS_1010_txt_number + ENUS_0930_txt_number + inner_txt_number + 4) == i:  # (0 + 14 + 10 +16 + 28 +4) 72
                            val_set.append(fea)
                            val_labels.append(onehot_label)
                        else:
                            train_set.append(fea)
                            train_labels.append(onehot_label)
                if len(val_set) == 0 and i < 84:
                    print("ENUS_1013 val_set size = 0, skip CV%d" % i)
                    continue
                print("after add ENUS_1013, train_set, val_set:", len(train_set), len(val_set))

    # read KOKR(1013) train data （CV 84 - 98）
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if filepath_label[8] in trainfiles_path:
                for KOKR_1013_txt_number in range(KOKR_1013_denoise_nums):  # 12 (KOKR_1010_denoise_nums: 0-11)
                    for item in data_set_KOKR_denoise_1013[KOKR_1013_txt_number]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        gender = name[1]
                        if gender not in gender_include:
                            continue
                        fea = [float(index) for index in temp[1:-1]]
                        emo_label = temp[0].strip("\'").split("_")[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if ( KOKR_1013_txt_number + ENUS_1013_txt_number + KOKR_1010_txt_number + ENUS_1010_txt_number + ENUS_0930_txt_number + inner_txt_number + 5) == i:  # (0 + 11+ 14 + 10 +16 + 28 +5) 72
                            val_set.append(fea)
                            val_labels.append(onehot_label)
                        else:
                            train_set.append(fea)
                            train_labels.append(onehot_label)
                if len(val_set) == 0 and i < 99:
                    print("KOKR_1013 val_set size = 0, skip CV%d" % i)
                    continue
                print("after add KOKR_1013, train_set, val_set:", len(train_set), len(val_set))

    # read spped data
    if len(train_set_path) != 0:
        for trainfiles_path in train_set_path:
            if  filepath_label[3] in trainfiles_path:                                  #  byperson_denoise_speed
                for speed_txt_num in range(set_num):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-3]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
            if  filepath_label[4] in trainfiles_path:                                   # ENUS_0930_denoise_speed
                for speed_txt_num in range(set_num,set_num+ENUS_0930_denoise_speed_nums):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-3]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
            if  filepath_label[5] in trainfiles_path:                               # ENUS_1010_denoise_speed
                for speed_txt_num in range(set_num+ENUS_0930_denoise_nums,set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-2]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
            '''
            if  filepath_label[6] in trainfiles_path:                               # KOKR_1010_denoise_speed
                for speed_txt_num in range(set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums,
                                           set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums+KOKR_1010_denoise_speed_nums):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-2]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
            '''
            if  filepath_label[9] in trainfiles_path:                               # ENUS_1013_denoise_speed
                for speed_txt_num in range(set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums+KOKR_1010_denoise_speed_nums,
                                           set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums+KOKR_1010_denoise_speed_nums+ENUS_1013_denoise_speed_nums):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-2]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
            if  filepath_label[10] in trainfiles_path:                               # KOKR_1013_denoise_speed
                for speed_txt_num in range(set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums+KOKR_1010_denoise_speed_nums+ENUS_1013_denoise_speed_nums,
                                           set_num+ENUS_0930_denoise_nums+ENUS_1010_denoise_spped_nums+KOKR_1010_denoise_speed_nums+ENUS_1013_denoise_speed_nums+KOKR_1013_denoise_speed_nums):
                    for item in data_set_speed[speed_txt_num]:
                        temp = item.strip().split(',')
                        name = temp[0].strip('\'').split('_')
                        fea = [float(index)for index in temp[1:-1]]
                        if 'fast' in temp[0] or 'slow' in temp[0]:
                            emo_label = temp[0].strip('\'').split('_')[-2]
                        else:
                            emo_label = temp[0].strip('\'').split('_')[-1]
                        if emo_label not in emo_classes.keys():
                            continue
                        onehot_label = np.zeros(class_num)
                        onehot_label[emo_classes[emo_label]] = 1
                        if speed_txt_num == i:           # 和 CV 相同的人不参与训练集
                            continue
                        train_set.append(fea)
                        train_labels.append(onehot_label)
    print("after add speed data train_set, val_set:", len(train_set), len(val_set))

    #add extra train data to train_set
    if len(extra_train_set_path)!=0 :
        for item in extra_train_set.data:
            temp=item.strip().split(',')
            name = temp[0].strip('\'').split('_')
            person=name[0][:2]
            if "Ses" in name[0]:
                fea = [float(index) for index in temp[1:-1]]
                #emo_label=temp[-1]
                emo_label=name[-1]
                if emo_label not in emo_classes.keys():
                    continue
                onehot_label = np.zeros(class_num)
                onehot_label[emo_classes[emo_label]] = 1
                train_set.append(fea)
                train_labels.append(onehot_label)
                continue
            if person in person_exclude:
                continue
            elif person.lstrip('0')==str(i+1):     # CV= 0 的时候， 加躁声中的01M 不进行训练， CV =1时， 加噪声中的02F不进行训练， 依次类推，到CV =20是 加躁声中21 不训练，但是加噪声中没有21, 所以从21开始后面的都将使用全部的加噪声
                continue
            gender = name[0][-1]
            if gender not in gender_include:
                continue
            scene=name[-3]
            if scene not in scene_include:
                continue

            scenario=name[-2]
            if  scenario not in scenario_include:
                continue
            db=name[-1]
            if db not in db_include:
                continue

            fea = [float(index) for index in temp[1:-1]]
            emo_label=temp[-1]
            if emo_label not in emo_classes.keys():
                continue
            onehot_label = np.zeros(class_num)
            onehot_label[emo_classes[emo_label]] = 1
            train_set.append(fea)
            train_labels.append(onehot_label)
    print("after add IEMO data train_set, val_set:", len(train_set), len(val_set))

    if len(val_set) == 0 and i < 72:
        print("KOKR_1010 val_set size = 0, skip CV%d" % i)
        continue
    exit()
    val_set = np.array(val_set)
    val_labels = np.array(val_labels)
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)

    # normalize
    # z-score
    mu = np.average(train_set, axis=0)
    variance = np.var(train_set, axis=0)

    '''start model training'''

    print("train_set, val_set:", len(train_set), len(val_set))
    print('Normalize:z-score')
    print('class_num:%d'%(class_num))
    print('Train_entry:%d' % len(train_set))
    print('Val_entry:%d' % len(val_set))
    print('epoch_num:%d'% epoch_num)
    print('batch_size:%d'%_batch_size)
    print('do_dropout:%d'%do_dropout)
    print('corpus:%s'%corpus)
    print('which_copy:%s'% which_copy)
    print('gender_include:%s' % (gender_include))
    print('person_exclude:%s'%(person_exclude))
    if len(extra_train_set_path)!=0 :
        print("extra_train_set:%s"%extra_train_set_path)
    if extra_val_set_path!="" :
        print("extra_val_set:%s"%extra_val_set_path)

    # 输出网络信息
    net_print = '------网络信息------\n'
    net_print += '网络类型:LSTM\n'
    net_print += '网络结构:'+str(fea_dim)+":"
    for i_t in hidden_size:
        net_print+=str(i_t)+":"
    net_print+=str(class_num)
    print(net_print)

    # 输出网络信息
    print('------额外训练集------\n')
    print('scene_include:%s'%(scene_include))
    print('scenario_include:%s'%(scenario_include))
    print('db_include:%s'%(db_include))


    # network define
    g = tf.Graph()
    with g.as_default():
        train_mean = tf.constant(mu, name="mu", dtype="float")
        train_var = tf.constant(variance, name="var", dtype="float")

        batch_size = tf.placeholder(tf.int32,name='batch_size')
        x = tf.placeholder('float', [None, fea_dim], name='input')
        y_ = tf.placeholder('float', [None, class_num], name='label')
        x_lengths=tf.placeholder('int32',[None],name='x_lengths')
        keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        # do z-socre
        x_normalized=tf.nn.batch_normalization(x, train_mean, train_var, 0, 2, 0.001, name="normalize")

        x_panding=tf.reshape(x_normalized,[1,-1,fea_dim])

        hidden_layer_num=len(hidden_size)
        hidden_layer=[]

        for h_l in range(hidden_layer_num):
            # add an LSTM layer
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size[h_l], forget_bias=1.0, state_is_tuple=True)
            # add dropout layer
            if do_dropout==1:
                lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            hidden_layer.append(lstm_cell)

        # 调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = rnn.MultiRNNCell(hidden_layer, state_is_tuple=True)

        # 用全零来初始化state

        init_state=mlstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, last_states=tf.nn.dynamic_rnn(cell=mlstm_cell,dtype=tf.float32,
        inputs= x_panding, initial_state=init_state, time_major=True, sequence_length=x_lengths)

        h_state=last_states[-1][-1]

        W = tf.Variable(tf.truncated_normal([hidden_size[-1], class_num], stddev=0.1), dtype=tf.float32,name='W_output')
        # W = tf.Variable(name='W_output', shape = [hidden_size[-1], class_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer)
        bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32,name='b_output')
        y = tf.nn.softmax(tf.matmul(h_state, W) + bias, name='predict')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        print("cost:", cost)
        print("optimizer:", optimizer)
        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # if tf.argmax(y, 1) = 0  ang的概率 # 计算出来的情感是否等于标记的标签情感，如果等于为true, 不相等为false
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    # 计算true占总体数据的平均值，就是占总体概率了

        # run epoch
        if save_model == 1:
            saver = tf.train.Saver()
        acc_val_max=-1

        train_data = CDataSet(train_set, train_labels)

        vali_data  = CDataSet(val_set, val_labels)
        vali_batch = vali_data.next_batch(len(val_set))

        with tf.Session() as sess:
            sess.run(init)
            avg_cost = []    # list
            acc_val = []
            acc_train = []

            vali_acc_fouremo_list = []
            train_acc_fouremo_list = []

            if save_model == 1:
                modelpath = cv_dir + '/modelinit.ckpt'
                saver.save(sess, modelpath, global_step=0)

            # 开始迭代训练神经网络， 总共迭代128轮
            for epoch in range(epoch_num):
                starttime = datetime.datetime.now()
                print("\nStrart Epoch %d traing:" % (epoch))
                if save_model == 1:
                    modelpath = cv_dir + '/model' + str(epoch) + '.ckpt'
                batch_num = 0
                batch_cost = 0
                if epoch >=len(_keep_prob):
                    keep_prob_index=-1
                else:
                    keep_prob_index=epoch
                train_right_array = np.zeros((4))
                train_total_array = np.zeros((4))
                vali_right_array = np.zeros((4))
                vali_total_array = np.zeros((4))
                while True:
                    batch = train_data.next_batch(_batch_size)
                    # print('\tbatch_num=%d,batch_size=%d'%(batch_num,batch[2]))
                    if batch[2] == 0:
                        break
                    _,c=sess.run([optimizer, cost], feed_dict={x: batch[0], y_:batch[1],
                    x_lengths:np.ones(batch[2], dtype='int'),batch_size:batch[2],keep_prob:_keep_prob[keep_prob_index]})
                    batch_num = batch_num + 1
                    batch_cost = (batch_num - 1) / batch_num * batch_cost + c / batch_num

                # calculate epoch accuracy of four emotions in vali_set
                vali_emo_predict = sess.run(y, feed_dict={x: vali_batch[0],
                                                     x_lengths: np.ones(vali_batch[2], dtype='int'),
                                                     batch_size: vali_batch[2],
                                                     keep_prob: 1.0})

                vali_label_prdct = np.argmax(vali_emo_predict, axis=1)

                vali_batch_y = np.argmax(vali_batch[1], axis=1)
                vali_length = vali_batch_y.shape[0]
                for vali_wav_order in range(vali_length):
                    vali_total_array[vali_batch_y[vali_wav_order]] += 1
                    if vali_batch_y[vali_wav_order] == vali_label_prdct[vali_wav_order]:
                        vali_right_array[vali_batch_y[vali_wav_order]] += 1
                print ("vali_right_array:", vali_right_array)
                print ("vali_total_array:", vali_total_array)
                vali_acc_array = vali_right_array/vali_total_array
                print("vali_acc_array", vali_acc_array)

                # calculate epoch accuracy of four emotions in train set
                train_emo_predict = sess.run(y, feed_dict={x: train_set,
                                                     x_lengths: np.ones(len(train_labels), dtype='int'),
                                                     batch_size: len(train_labels),
                                                     keep_prob: 1.0})
                train_label_prdct = np.argmax(train_emo_predict,
                                        axis=1)
                train_batch_y = np.argmax(train_labels,
                                    axis=1)
                length = train_batch_y.shape[0]
                for train_wav_order in range(length):
                    train_total_array[train_batch_y[train_wav_order]] += 1
                    if train_batch_y[train_wav_order] == train_label_prdct[train_wav_order]:
                        train_right_array[train_batch_y[train_wav_order]] += 1
                print ("train_right_array:", train_right_array)
                print ("train_total_array:", train_total_array)
                train_acc_array = train_right_array/train_total_array
                print("train_acc_array", train_acc_array)

                # 将每类的情感概率加入追加到列表中， 列表中的每个元素都是4个数的列表
                vali_acc_fouremo_list.append(vali_acc_array)
                train_acc_fouremo_list.append(train_acc_array)

                avg_cost.append(batch_cost)
                acc_val.append(sess.run(accuracy, feed_dict={x: val_set, y_: val_labels, x_lengths :np.ones(len(val_labels),dtype='int'),batch_size:len(val_labels),keep_prob:1.0}))
                acc_train.append(sess.run(accuracy, feed_dict={x:train_set, y_: train_labels, x_lengths:np.ones(len(train_labels),dtype='int'),batch_size:len(train_labels),keep_prob:1.0}))
                print('Epoch %d finished' % (epoch))
                print('\tavg_cost = %f' % (avg_cost[epoch]))
                print('\tacc_train = %f' % (acc_train[epoch]))
                print('\tacc_val = %f' % (acc_val[epoch]))
                if save_model == 1:
                   if acc_val_max < acc_val[epoch]:
                   #if epoch %2==0 :
                        rt = saver.save(sess, modelpath)
                        acc_val_max = acc_val[epoch]
                        val_test_emo_predict = sess.run(y, feed_dict={x: val_set,
                                                                   x_lengths: np.ones(len(val_labels),
                                                                                      dtype='int'),
                                                                   batch_size: len(val_labels),
                                                                   keep_prob: 1.0})  # 测试集上模型预测得到的结果，是一个len(test_labels)行 * 7 列的列表

                        mix_folder = cv_dir + "/" + str(epoch)
                        mix_matrix(val_labels, val_test_emo_predict, mix_folder)
                        print('model saved in %s' % (rt))
                        print('packing model to .pb format')

                if acc_train[epoch] > acc_train_epsilon:              # 训练集 大于 0.98
                    break
                endtime = datetime.datetime.now()
                print("seconds:", (endtime-starttime).seconds)

        # calculate CV accuracy of four emotions
        cv_vali_acc_fouremo_list.append(vali_acc_fouremo_list[np.argmax(np.mean(vali_acc_fouremo_list, axis=1))])
        cv_train_acc_fouremo_list.append(train_acc_fouremo_list[np.argmax(np.mean(train_acc_fouremo_list, axis =1))])

        # calculate CV accuracy
        cv_vali_acc.append(acc_val[np.argmax(acc_val)])
        cv_train_acc.append(acc_train[np.argmax(acc_val)])

        print("每个CV valitation set的四类情感的识别率， 是一个1行4列的列表： cv %d 准确率：" % (i))
        print(cv_vali_acc_fouremo_list[-1])
        print("每个CV train set的四类情感的识别率， 是一个1行4列的列表： cv %d 准确率：" % (i))
        print(cv_train_acc_fouremo_list[-1])

        print('cv_vali_acc on cv %d = %f' % (i, cv_vali_acc[-1]))  # 返回每次cv时的训练准确率
        print('cv_train_acc on cv %d = %f' % (i, cv_train_acc[-1]))  # 返回每次cv时的验真准确率

    print('CV %d finish ' % (i))
print('--------------------------------------------------------')
print('training finished. SRCB meetingroom all average:\ncv_train_acc=%f,cv_vali_acc=%f' % (np.average(cv_train_acc[0:21]), np.average(cv_vali_acc[0:21])))   #
print('training finished. SRCB office all average: \ncv_train_acc=%f. cv_vali_acc=%f' % (np.average(cv_train_acc[21:27]), np.average(cv_vali_acc[21:27])))
print('training finished. SRCB shoppingmall all average: \ncv_train_acc=%f,cv_vali_acc=%f' % (np.average(cv_train_acc[27:29]), np.average(cv_vali_acc[27:29])))

print('training finished. ENUS_0930_M all average: \navg_train_cv_ENUS_M=%f,avg_val_cv_ENUS_M=%f' % (np.average(cv_train_acc[29:35]), np.average(cv_vali_acc[29:35])))
print('training finished. ENUS_0930_O all average: \navg_train_cv_ENUS_0=%f,avg_val_cv_ENUS_0=%f' % (np.average(cv_train_acc[35:41]), np.average(cv_vali_acc[35:41])))
print('training finished. ENUS_0930_S all average: \navg_train_cv_ENUS_S=%f,avg_val_cv_ENUS_S=%f' % (np.average(cv_train_acc[41:46]), np.average(cv_vali_acc[41:46])))

print('training finished. ENUS_1010_M all average: \navg_train_cv_ENUS_M=%f,avg_val_cv_ENUS_M=%f' % (np.average(cv_train_acc[46:51]), np.average(cv_vali_acc[46:51])))
print('training finished. ENUS_1010_O all average: \navg_train_cv_ENUS_0=%f,avg_val_cv_ENUS_0=%f' % (np.average(cv_train_acc[51:55]), np.average(cv_vali_acc[51:55])))
print('training finished. ENUS_1010_S all average: \navg_train_cv_ENUS_S=%f,avg_val_cv_ENUS_S=%f' % (np.average(cv_train_acc[55:57]), np.average(cv_vali_acc[55:57])))

print('training finished. KOKR_1010_M all average: \navg_train_cv_KOKR_M=%f,avg_val_cv_KOKR_M=%f' % (np.average(cv_train_acc[57:62]), np.average(cv_vali_acc[57:62])))
print('training finished. KOKR_1010_O all average: \navg_train_cv_KOKR_0=%f,avg_val_cv_KOKR_0=%f' % (np.average(cv_train_acc[62:67]), np.average(cv_vali_acc[62:67])))
print('training finished. KOKR_1010_S all average: \navg_train_cv_KOKR_S=%f,avg_val_cv_KOKR_S=%f' % (np.average(cv_train_acc[67:72]), np.average(cv_vali_acc[67:72])))

print('training finished. ENUS_1013_M all average: \navg_train_cv_ENUS_M=%f,avg_val_cv_ENUS_M=%f' % (np.average(cv_train_acc[72:76]), np.average(cv_vali_acc[72:76])))
print('training finished. ENUS_1013_O all average: \navg_train_cv_ENUS_0=%f,avg_val_cv_ENUS_0=%f' % (np.average(cv_train_acc[76:80]), np.average(cv_vali_acc[76:80])))
print('training finished. ENUS_1013_S all average: \navg_train_cv_ENUS_S=%f,avg_val_cv_ENUS_S=%f' % (np.average(cv_train_acc[80:84]), np.average(cv_vali_acc[80:84])))

print('training finished. KOKR_1013_M all average: \navg_train_cv_KOKR_M=%f,avg_val_cv_KOKR_M=%f' % (np.average(cv_train_acc[84:89]), np.average(cv_vali_acc[84:89])))
print('training finished. KOKR_1013_O all average: \navg_train_cv_KOKR_0=%f,avg_val_cv_KOKR_0=%f' % (np.average(cv_train_acc[89:94]), np.average(cv_vali_acc[89:94])))
print('training finished. KOKR_1013_S all average: \navg_train_cv_KOKR_S=%f,avg_val_cv_KOKR_S=%f' % (np.average(cv_train_acc[94:99]), np.average(cv_vali_acc[94:99])))       # （CV 84 - 98）

print("***************************************************************")
print("SRCB meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[0:21]), np.mean(cv_vali_acc_fouremo_list[0:21], axis = 0))
print("SRCB office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[21:27]), np.mean(cv_vali_acc_fouremo_list[21:27], axis = 0))
print("SRCB shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[27:29]), np.mean(cv_vali_acc_fouremo_list[27:29], axis = 0))

print("ENUS_0930_M meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[29:35]), np.mean(cv_vali_acc_fouremo_list[29:35], axis = 0))
print("ENUS_0930_O office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[35:41]), np.mean(cv_vali_acc_fouremo_list[35:41], axis = 0))
print("ENUS_0930_S shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[41:46]), np.mean(cv_vali_acc_fouremo_list[41:46], axis = 0))

print("ENUS_1010_M meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[46:51]), np.mean(cv_vali_acc_fouremo_list[46:51], axis = 0))
print("ENUS_1010_O office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[51:55]), np.mean(cv_vali_acc_fouremo_list[51:55], axis = 0))
print("ENUS_1010_S shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[55:57]), np.mean(cv_vali_acc_fouremo_list[55:57], axis = 0))

print("KOKR_1010_M meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[57:62]), np.mean(cv_vali_acc_fouremo_list[57:62], axis = 0))
print("KOKR_1010_O office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[62:67]), np.mean(cv_vali_acc_fouremo_list[62:67], axis = 0))
print("KOKR_1010_S shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[67:72]), np.mean(cv_vali_acc_fouremo_list[67:72], axis = 0))

print("ENUS_1010_M meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[72:76]), np.mean(cv_vali_acc_fouremo_list[72:76], axis = 0))
print("ENUS_1010_O office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[76:80]), np.mean(cv_vali_acc_fouremo_list[76:80], axis = 0))
print("ENUS_1010_S shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[80:84]), np.mean(cv_vali_acc_fouremo_list[80:84], axis = 0))

print("KOKR_1010_M meetingroom cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[84:89]), np.mean(cv_vali_acc_fouremo_list[84:89], axis = 0))
print("KOKR_1010_O office cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[89:94]), np.mean(cv_vali_acc_fouremo_list[89:94], axis = 0))
print("KOKR_1010_S shoppingmall cv_vali_acc_fouremo_list:", len(cv_vali_acc_fouremo_list[94:99]), np.mean(cv_vali_acc_fouremo_list[94:99], axis = 0))

if output_log == 1:
    # 恢复标准输入输出
    sys.stdout = savedStdout
    fin.close()
print('Training finished.')


if output_log == 1:
    # 恢复标准输入输出
    sys.stdout = savedStdout
    fin.close()
print('Training finished.')

