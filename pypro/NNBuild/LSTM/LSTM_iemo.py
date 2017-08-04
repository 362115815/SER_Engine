#!/usr/bin/env python
# encoding: utf-8

import numpy as np 
import random 
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import sys
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '8'

'''
This is an LSTM network building script
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


#config


output_log=0
save_model=0
gender_include = 'M,F'
timestep_size = 1
corpus ='iemo'
which_copy='byperson'
do_dropout=1
_keep_prob=[0.5,0.5,0.2,0.1]

# 每个隐含层的节点数
hidden_size = 512
# LSTM layer 的层数
layer_num = 1
acc_train_epsilon= 0.9
epoch_num = 256
_batch_size=256
learning_rate = 0.001
hidden_layer=1
# predefine

set_num = 10
fea_dim = 88
emo_classes = {'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3}


class_num = len(emo_classes)-1

now = datetime.datetime.now()

if output_log == 1:
    # 标准输入输出重定向到文本
    savedStdout = sys.stdout
    fin = open(logdir + '/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.log', 'w+')
    sys.stdout = fin

print('*********************************************')
print('******* Run LSTM %s ********' % (now.strftime('%Y-%m-%d %H:%M:%S')))
print('*********************************************')


# read data
data_set = []
for i in range(set_num):
    filepath = feadir+'/'+corpus+ '/' +which_copy+'/'+ str(i) + '.txt'
    with open(filepath, 'r') as fin:
        data_set.append(fin.readlines())

# CV

acc_val_cv = np.zeros(set_num)  # 每次CV 时val_set的准确率
acc_train_cv = np.zeros(set_num)  # 每次CV 时train_set的准确率

cv_now = datetime.datetime.now()

for i in range(set_num):
    print('Begin CV %d :' % (i))
    if save_model==1:
        cv_dir = modeldir + '/' + cv_now.strftime('%Y-%m-%d_%H_%M_%S') + '_cv' + str(i)
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
    print('Gender_include:%s' % (gender_include))
    print('epoch_num:%d'% epoch_num)
    print('batch_size:%d'%_batch_size)
    print('do_dropout:%d'%do_dropout)
    # 输出网络信息
    net_print = '------网络信息------\n'
    net_print += '网络类型:LSTM\n'
    net_print += '网络结构:'+str(fea_dim)+":"+str(hidden_size)+":"+str(class_num)
    print(net_print)



    # network define
    g = tf.Graph()
    with g.as_default():

        train_mean = tf.constant(mu, name="mu", dtype="float")
        train_var = tf.constant(variance, name="var", dtype="float")

        batch_size = tf.placeholder(tf.int32) 
        x = tf.placeholder('float', [None, fea_dim], name='input')
        y_ = tf.placeholder('float', [None, class_num], name='label')
        x_lengths=tf.placeholder('int32',[None])
        keep_prob=tf.placeholder(tf.float32)
        # do z-socre
        x_normalized = tf.nn.batch_normalization(x, train_mean, train_var, 0, 2, 0.001, name="normalize")



        x_panding=tf.Variable(0,validate_shape=False,dtype=tf.float32)

        x_panding=tf.assign(x_panding,[x_normalized],validate_shape= False)


        # add an LSTM layer
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

        # add dropout layer
        if do_dropout==1:
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

        # 调用 MultiRNNCell 来实现多层 LSTM
        # 
        mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

        # 用全零来初始化state
        
        init_state=mlstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, last_states=tf.nn.dynamic_rnn(cell=mlstm_cell,dtype=tf.float32,
        inputs= x_panding, initial_state=init_state, time_major=True, sequence_length=x_lengths)

        h_state=last_states[0][1]

        W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
        y = tf.nn.softmax(tf.matmul(h_state, W) + bias, name='predict')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

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
                   # print(batch[2])
                   # o,h = sess.run([outputs, last_states], feed_dict={x: batch[0], y_:batch[1],
                   #  x_lengths:np.ones(batch[2], dtype='int'),batch_size:[batch[2]]})
                    _,c=sess.run([optimizer, cost], feed_dict={x: batch[0], y_:batch[1],
                     x_lengths:np.ones(batch[2], dtype='int'),batch_size:batch[2],keep_prob:_keep_prob[keep_prob_index]})
                    batch_num = batch_num + 1
                    batch_cost = (batch_num - 1) / batch_num * batch_cost + c / batch_num
                avg_cost.append(batch_cost)
                acc_val.append(sess.run(accuracy, feed_dict={x: val_set, y_: val_labels, x_lengths :np.ones(len(val_labels),dtype='int'),batch_size:len(val_labels),keep_prob:1.0}))
                acc_train.append(sess.run(accuracy, feed_dict={x: train_set, y_: train_labels, x_lengths:np.ones(len(train_labels),dtype='int'),batch_size:len(train_labels),keep_prob:1.0}))
                print('Epoch %d finished' % (epoch))
                print('\tavg_cost = %f' % (avg_cost[epoch]))
                print('\tacc_train = %f' % (acc_train[epoch]))
                print('\tacc_val = %f' % (acc_val[epoch]))
                if save_model == 1:
                    if acc_val_max < acc_val[epoch]:
                        rt = saver.save(sess, modelpath)
                        acc_val_max = acc_val[epoch]
                        print('model saved in %s' % (rt))
                if acc_train[epoch] > acc_train_epsilon:
                    break
        acc_val_cv[i] = acc_val[np.argmax(acc_val)]
        acc_train_cv[i] = acc_train[np.argmax(acc_val)]

    print('acc_train on cv %d = %f' % (i, acc_train_cv[i]))
    print('acc_val on cv %d = %f' % (i, acc_val_cv[i]))

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


















