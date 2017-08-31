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

os.environ['CUDA_VISIBLE_DEVICES'] = '11'

'''
This is a CNN network building script
'''

class CDataSet():
    def __init__(self, data, labels, shuffle=False):
        if len(data) == 0 or len(data) != len(labels):
            raise ValueError('data为空或data与label长度不匹配')
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self.is_shuffle = shuffle

    def _shuffle(self):
        c = list(zip(self.data, self.labels))
        random.shuffle(c)
        self.data[:], self.labels[:] = zip(*c)

    def next_batch(self, batch_size):  # 如果到达末尾，则把batch_size返回0，否则返回所读取的batch_size
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            return [], [], 0
        if (self.batch_id == 0):
            if self.is_shuffle == True:
                self._shuffle()
        end_id = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_id])
        batch_labels = (self.labels[self.batch_id:end_id])
        num=end_id-self.batch_id
        self.batch_id = end_id
        return batch_data, batch_labels,num 


# setting path

rootdir = '/data/mm0105.chen/wjhan/xiaomin'
feadir = rootdir + '/feature'
logdir = rootdir + '/log'
modeldir = rootdir + '/model'

extra_train_set_path=""#"/data/mm0105.chen/wjhan/xiaomin/feature/iemo/washedS8/iemo.arff"
extra_val_set_path="" #"/data/mm0105.chen/wjhan/xiaomin/feature/intern_noise/intern_noise_all.arff"

#config
output_log=1
save_model=1

corpus ='intern'
which_copy='byperson'
do_dropout=0
_keep_prob=[0.5,0.5,0.5,0.5,0.2,0.1]

gender_include = ['M','F']

person_exclude=['03','06','07','09','18','14']
scene_include=['office']
scenario_include=['meet','white','seat']
db_include=['25']


acc_train_epsilon= 0.98
epoch_num = 256
_batch_size=256
learning_rate = 0.0005


# predefine

set_num = 20
fea_dim = 88
emo_classes = {'ang': 0, 'hap': 1, 'nor': 2, 'sad': 3,'neu':2,'exc':1}

class_num =4

now = datetime.datetime.now()

if output_log == 1:
    # 标准输入输出重定向到文本
    log_path=logdir + '/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.log'
    print("redirect output to %s"%log_path)
    savedStdout = sys.stdout
    fin = open(log_path, 'w+')
    sys.stdout = fin

print('*********************************************')
print('******** Run CNN %s ********' % (now.strftime('%Y-%m-%d %H:%M:%S')))
print('*********************************************')




#读入额外训练集
if extra_train_set_path!="" :
    extra_train_set=dr.ArffReader(extra_train_set_path)

#读入额外验证集
if extra_val_set_path!="" :
    extra_val_set=dr.ArffReader(extra_val_set_path)

# read data
data_set = []
for i in range(set_num):
    filepath = feadir+'/'+corpus+ '/' +which_copy+'/'+ str(i) + '.txt'
    with open(filepath, 'r') as fin:
        data_set.append(fin.readlines())

# CV

acc_val_cv = [] # 每次CV 时val_set的准确率
acc_train_cv = []  # 每次CV 时train_set的准确率

for i in range(set_num):
    print('Begin CV %d :' % (i))
    if save_model==1:
        cv_dir = modeldir + '/' + now.strftime('%Y-%m-%d_%H_%M_%S') + '_cv' + str(i)
        os.system('mkdir ' + cv_dir)

    '''data prepare'''

    # create val_set,val_label,train_set,train_label
    val_set = []
    val_labels = []
    train_set = []
    train_labels = []
    for j in range(set_num):
        for item in data_set[j]:
            name = item.lstrip("\'").split("_", 1)[0]
            temp = item.split(',')
            gender = name[-1]
            name=name[:2]
            if name in person_exclude:
                continue
            if gender not in gender_include:
                continue
            fea = [float(index) for index in temp[1:-1]]
            emo_label=temp[-1].strip()
            if emo_label not in emo_classes.keys():
                continue
            onehot_label = np.zeros(class_num)
            onehot_label[emo_classes[emo_label]] = 1
            if j == i:
                val_set.append(fea)
                val_labels.append(onehot_label)
            else:

                train_set.append(fea)
                train_labels.append(onehot_label)
    if len(val_set)==0:
        print('val_set size=0 ,skip CV%d'%i)
        continue
    #add extra train data to train_set

    #额外训练集加入到train_set
    if extra_train_set_path!="" :
        for item in extra_train_set.data:
            temp=item.strip().split(',')
            name = temp[0].strip('\'').split('_')
            person=name[0][:2]
            if person in person_exclude:
                continue
            elif person.lstrip('0')==str(i+1):
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


    #额外验证集加入到val_set
    if extra_val_set_path!="" :
        for item in extra_val_set.data:
            temp=item.strip().split(',')
            name = temp[0].strip('\'').split('_')
            person=name[0][:2]
            if person in person_exclude:
                continue
            elif person.lstrip('0')!=str(i+1):
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
            val_set.append(fea)
            val_labels.append(onehot_label)
   

    val_set = np.array(val_set)
    val_labels = np.array(val_labels)
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)


    # normalize
    # z-score
    mu = np.average(train_set, axis=0)
    variance = np.var(train_set, axis=0)

    '''start model training'''


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
    if extra_train_set_path!="" :
        print("extra_train_set:%s"%extra_train_set_path)
    if extra_val_set_path!="" :
        print("extra_val_set:%s"%extra_val_set_path)



    # 输出网络信息
    net_print = '------网络信息------\n'
    net_print += '网络类型:CNN\n'
    net_print += '网络结构:'+str(fea_dim)+":\n"
    '''
    for i_t in hidden_size:
        net_print+=str(i_t)+":"
    net_print+=str(class_num)
    print(net_print)
    '''
    # 输出额外训练集信息
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


        keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        # do z-socre
        # 
        x_normalized=tf.nn.batch_normalization(x, train_mean, train_var, 0, 2, 0.001, name="normalize")
        x_reshape=tf.reshape(x_normalized,[-1,fea_dim,1])
        
        ### CONV 1###
        conv1_filters=32
        conv1_kernel_size=[3]
        conv1_strides=[1]
        pool1_size=[2]
        pool1_strides=[2]

        net_print+="conv1:\n"
        net_print+="/////////////////\n"
        net_print+="filters:"+str(conv1_filters)+"\n"+\
        "kernel_size:"+str(conv1_kernel_size)+"\n"+"strides:"+str(conv1_strides)+"\n"

        net_print+="pool1:\n"
        net_print+="pool method:max\n"
        net_print+="pool size:"+str(pool1_size)+"\n"
        net_print+="pool strides:"+str(pool1_strides)+"\n"
        net_print+="/////////////////\n"
 

        h_conv1=tf.layers.conv1d(x_reshape,filters=conv1_filters,kernel_size=conv1_kernel_size,strides=conv1_strides,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        h_pool1=tf.layers.max_pooling1d(h_conv1,pool_size=pool1_size,strides=pool1_strides)


        
        ### CONV 2 ###   
        conv2_filters=48

        conv2_kernel_size=[3]

        conv2_strides=[1]

        pool2_size=[2]

        pool2_strides=[2]

        net_print+="conv2:\n"
        net_print+="/////////////////\n"
        net_print+="filters:"+str(conv2_filters)+"\n"+\
        "kernel_size:"+str(conv2_kernel_size)+"\n"+"strides:"+str(conv2_strides)+"\n"

        net_print+="pool2:\n"
        net_print+="pool method:max\n"
        net_print+="pool size:"+str(pool2_size)+"\n"
        net_print+="pool strides:"+str(pool2_strides)+"\n"
        net_print+="/////////////////\n"


        h_conv2=tf.layers.conv1d(h_pool1,filters=conv2_filters,kernel_size=conv2_kernel_size,strides=conv2_strides,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        h_pool2=tf.layers.max_pooling1d(h_conv2,pool_size=pool2_size,strides=pool2_strides)
             
               
        ##### CONV 3 ######

        conv3_filters=64

        conv3_kernel_size=[3]

        conv3_strides=[1]

        pool3_size=[2]

        pool3_strides=[2]

        net_print+="conv3:\n"
        net_print+="/////////////////\n"
        net_print+="filters:"+str(conv3_filters)+"\n"+\
        "kernel_size:"+str(conv3_kernel_size)+"\n"+"strides:"+str(conv3_strides)+"\n"

        net_print+="pool3:\n"
        net_print+="pool method:max\n"
        net_print+="pool size:"+str(pool3_size)+"\n"
        net_print+="pool strides:"+str(pool3_strides)+"\n"
        net_print+="/////////////////\n"


        h_conv3=tf.layers.conv1d(h_pool2,filters=conv3_filters,kernel_size=conv3_kernel_size,strides=conv3_strides,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        h_pool3=tf.layers.max_pooling1d(h_conv3,pool_size=pool3_size,strides=pool3_strides)
        

               
        ##### CONV 4 ######

        conv4_filters=96

        conv4_kernel_size=[3]

        conv4_strides=[1]

        pool4_size=[2]

        pool4_strides=[2]

        net_print+="conv4:\n"
        net_print+="/////////////////\n"
        net_print+="filters:"+str(conv4_filters)+"\n"+\
        "kernel_size:"+str(conv4_kernel_size)+"\n"+"strides:"+str(conv4_strides)+"\n"

        net_print+="pool4:\n"
        net_print+="pool method:max\n"
        net_print+="pool size:"+str(pool4_size)+"\n"
        net_print+="pool strides:"+str(pool4_strides)+"\n"
        net_print+="/////////////////\n"


        h_conv4=tf.layers.conv1d(h_pool3,filters=conv4_filters,kernel_size=conv4_kernel_size,strides=conv4_strides,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        h_pool4=tf.layers.max_pooling1d(h_conv4,pool_size=pool4_size,strides=pool4_strides)
        
        ##### CONV 5 ######

        conv5_filters=128

        conv5_kernel_size=[3]

        conv5_strides=[1]

        pool5_size=[2]

        pool5_strides=[2]

        net_print+="conv5:\n"
        net_print+="/////////////////\n"
        net_print+="filters:"+str(conv5_filters)+"\n"+\
        "kernel_size:"+str(conv5_kernel_size)+"\n"+"strides:"+str(conv5_strides)+"\n"

        net_print+="pool5:\n"
        net_print+="pool method:max\n"
        net_print+="pool size:"+str(pool5_size)+"\n"
        net_print+="pool strides:"+str(pool5_strides)+"\n"
        net_print+="/////////////////\n"


        h_conv5=tf.layers.conv1d(h_pool4,filters=conv5_filters,kernel_size=conv5_kernel_size,strides=conv5_strides,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        h_pool5=tf.layers.max_pooling1d(h_conv5,pool_size=pool5_size,strides=pool5_strides)

        h_flat=tf.contrib.layers.flatten(h_pool5)

        h_full1=tf.contrib.layers.fully_connected(h_flat,1024,activation_fn=tf.nn.softmax)
        #h_full2=tf.contrib.layers.fully_connected(h_full1,512,activation_fn=tf.nn.softmax)

        y=tf.contrib.layers.fully_connected(h_full1,class_num,activation_fn=tf.nn.softmax)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print(net_print)



        # run epoch
        if save_model == 1:
            saver = tf.train.Saver()
        acc_val_max=-1
        train_data = CDataSet(train_set, train_labels)

        with tf.Session() as sess:
            sess.run(init)

            avg_cost = []
            acc_val = []
            acc_train = []
            if save_model == 1:
                modelpath = cv_dir + '/modelinit.ckpt'
                saver.save(sess, modelpath, global_step=0)

            for epoch in range(epoch_num):
                print("\nStrart Epoch %d traing:" % (epoch))
                if save_model == 1:
                    modelpath = cv_dir + '/model' + str(epoch) + '.ckpt'
                batch_num = 0
                batch_cost = 0
                if epoch >=len(_keep_prob):
                    keep_prob_index=-1
                else:
                    keep_prob_index=epoch

                while True:
                    batch = train_data.next_batch(_batch_size)
                    # print('\tbatch_num=%d,batch_size=%d'%(batch_num,batch[2]))
                    if batch[2] == 0:
                        break
                    '''
                    o2,o1,o= sess.run([x_reshape,h_pool1,h_conv1], feed_dict={x: batch[0], y_:batch[1],
                    batch_size:batch[2],keep_prob:_keep_prob[keep_prob_index]})
                    print('x_reshape=')
                    #print(o2)
                    print(o2.shape)

                    print('h_conv1=\n')
                    #print(o)
                    print(o.shape)
                    print('h_pool1=\n')
                    #print(o1)
                    print(o1.shape)
                    exit()
                    '''
                    '''
                    o1,o2=sess.run([h_flat,h_full1],feed_dict={x:batch[0],y_:batch[1],batch_size:batch[2],keep_prob:_keep_prob[keep_prob_index]})

                    print('h_hull1 shape:')
                    print(o2.shape)
                    print('\n')
                    print('h_flat shape:')
                    print(o1.shape)
                    print('\n')
                    exit()
                    '''

                    _,c=sess.run([optimizer, cost], feed_dict={x: batch[0], y_:batch[1],
                    batch_size:batch[2],keep_prob:_keep_prob[keep_prob_index]})

                    batch_num = batch_num + 1
                    batch_cost = (batch_num - 1) / batch_num * batch_cost + c / batch_num
                avg_cost.append(batch_cost)
                acc_val.append(sess.run(accuracy, feed_dict={x: val_set, y_: val_labels,batch_size:len(val_labels),keep_prob:1.0}))
                acc_train.append(sess.run(accuracy, feed_dict={x:train_set, y_: train_labels,batch_size:len(train_labels),keep_prob:1.0}))
                print('Epoch %d finished' % (epoch))
                print('\tavg_cost = %f' % (avg_cost[epoch]))
                print('\tacc_train = %f' % (acc_train[epoch]))
                print('\tacc_val = %f' % (acc_val[epoch]))
                if save_model == 1:
                    if acc_val_max < acc_val[epoch]:
                        rt = saver.save(sess, modelpath)
                        acc_val_max = acc_val[epoch]
                        print('model saved in %s' % (rt))
                        print('packing model to .pb format')

                if acc_train[epoch] > acc_train_epsilon:
                    break
        acc_val_cv.append(acc_val[np.argmax(acc_val)])
        acc_train_cv.append( acc_train[np.argmax(acc_val)])

    print('acc_train on cv %d = %f' % (i, acc_train_cv[-1]))
    print('acc_val on cv %d = %f' % (i, acc_val_cv[-1]))

    print('CV %d finish ' % (i))
print('CV finished.\navg_acc_train=%f,avg_acc_val=%f' % (np.average(acc_train_cv), np.average(acc_val_cv)))

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


