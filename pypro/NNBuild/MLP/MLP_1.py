#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import random
import tensorflow as tf
import os
import sys
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '8'


def BatchNorm(value, is_train=True, epsilon=1e-5, momentum=0.9):
    return tf.contrib.layers.batch_norm(value, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True,
                                        is_training=is_train)


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
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_size


# setting path

rootdir = 'D:/xiaomin'
feadir = rootdir + '/feature'
logdir = rootdir + '/log'
modeldir = rootdir + '/model'


emo_classes = {'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'fru': 3, 'sad': 4}


'''
# read gender info
# 读入性别信息
fin = open(rootdir + "/feature/speakerInfo.txt", "r")
speakerInfo = fin.readlines()
fin.close()
genderInfo = {}
for item in speakerInfo:
    item = item.rstrip("\r\n")
    name, gender = item.split("\t")
    genderInfo[name] = gender
'''


# config
no_paragraph = 1
gender_include = 'M,F'
output_log = 0
save_model = 0
do_BN = 0
bn_decay = 0.9
bn_beta = 0
bn_gamma = 5
acc_train_epsilon = 0.95

now = datetime.datetime.now()

if output_log == 1:
    # 标准输入输出重定向到文本
    savedStdout = sys.stdout
    fin = open(logdir + '/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.log', 'w+')
    sys.stdout = fin

print('*********************************************')
print('****** Run MLP.py %s *******' % (now.strftime('%Y-%m-%d %H:%M:%S')))
print('*********************************************')

data_set = []
set_num = 10
fea_dim = 88
class_num = len(emo_classes)

for i in range(set_num):
    filepath = feadir+'/iemo' + '/' + str(i) + '.txt'
    with open(filepath, 'r') as fin:
        data_set.append(fin.readlines())

# CV

acc_val_cv = np.zeros(set_num)  # 每次CV 时val_set的准确率
acc_train_cv = np.zeros(set_num)  # 每次CV 时train_set的准确率

cv_now = datetime.datetime.now()

for i in range(set_num):
    print('Begin CV %d :' % (i))
    cv_dir = modeldir + '/' + cv_now.strftime('%Y-%m-%d_%H_%M_%S') + '_cv' + str(i)
    os.system('mkdir ' + cv_dir)

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
            emo_label=temp[-1].replace('\n', '')
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
    # start model training

    val_set = np.array(val_set)
    val_labels = np.array(val_labels)
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)

    epoch_num = 1024
    batch_size = 512
    learning_rate = 0.001
    hidden_layer = [1024]

    hidden_layer_num = len(hidden_layer)
    in_node_num = fea_dim
    out_node_num = class_num

    W = []
    b = []

    # normalize
    # z-score
    mu = np.average(train_set, axis=0)
    variance = np.var(train_set, axis=0)

    # train_set=(train_set-mu)/sigma
    # val_set=(val_set-mu)/sigma

    print('Normalize:z-score')

    print('Train_entry:%d' % len(train_set))
    print('Val_entry:%d' % len(val_set))
    print('Batch_size:%d' % (batch_size))
    print('Gender_include:%s' % (gender_include))
    if no_paragraph == 1:
        print('no_paragraph:True')
    # define neural net
    g = tf.Graph()
    with g.as_default():
        train_mean = tf.constant(mu, name="mu", dtype="float")
        train_var = tf.constant(variance, name="var", dtype="float")
        x = tf.placeholder('float', [None, fea_dim], name='input')
        y_ = tf.placeholder('float', [None, class_num], name='label')
        is_train = tf.placeholder(tf.bool)

        x_normalized = tf.nn.batch_normalization(x, train_mean, train_var, 0, 2, 0.001, name="normalize")
        net_struct = []
        net_struct.append(in_node_num)

        for item in hidden_layer:
            net_struct.append(item)

        net_struct.append(out_node_num)

        # 输出网络信息
        net_print = '------网络信息------\n'
        net_print += '网络类型:MLP\n'
        net_print += '网络结构:'

        for net_struct_it in range(len(net_struct) - 1):
            net_print += str(net_struct[net_struct_it])
            net_print += ':'
        net_print += str(net_struct[-1])
        net_print += '\n--------------------'
        print(net_print)
        print('do_BN:%d' % (do_BN))
        net_depth = len(net_struct)

        for index in range(1, net_depth):
            W.append(tf.Variable(tf.random_normal([net_struct[index - 1], net_struct[index]])))
            b.append(tf.Variable(tf.random_normal([net_struct[index]])))

        layer_out = list(range(net_depth))
        print(type(layer_out))
        print(layer_out)
        layer_out[0] = x_normalized

        for index in range(1, net_depth - 1):
            layer_out[index] = tf.add(tf.matmul(layer_out[index - 1], W[index - 1]), b[index - 1])
            if do_BN == 1:  # 添加BN层
                layer_out[index] = BatchNorm(layer_out[index], is_train=is_train)
            layer_out[index] = tf.nn.sigmoid(layer_out[index])

        layer_out[-1] = tf.add(tf.matmul(layer_out[-2], W[-1]), b[-1])

        if do_BN == 1:
            layer_out[-1] = BatchNorm(layer_out[-1], is_train=is_train)

        y = tf.nn.softmax(layer_out[-1], name='predict')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # run epoch
        if save_model == 1:
            saver = tf.train.Saver()
        acc_val_max = -1
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
                while True:
                    batch = train_data.next_batch(batch_size)

                    # print('\tbatch_num=%d,batch_size=%d'%(batch_num,batch[2]))
                    if batch[2] == 0:
                        break
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y_: batch[1], is_train: True})
                    batch_num = batch_num + 1
                    batch_cost = (batch_num - 1) / batch_num * batch_cost + c / batch_num
                avg_cost.append(batch_cost)
                acc_val.append(sess.run(accuracy, feed_dict={x: val_set, y_: val_labels, is_train: False}))
                acc_train.append(sess.run(accuracy, feed_dict={x: train_set, y_: train_labels, is_train: False}))
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


















