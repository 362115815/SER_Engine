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

#global config
out_data=0
output_log=1
save_model=1

os.environ['CUDA_VISIBLE_DEVICES'] = '11'
'''
This is an LSTM network building script
'''

class CDataSet():
    def __init__(self, data, labels, shuffle=True):
        if len(data) == 0 or len(data) != len(labels):
            raise ValueError('data为空或data与label长度不匹')
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self.is_shuffle = shuffle

    def _shuffle(self):
        c = list(zip(self.data, self.labels))
        random.shuffle(c)
        self.data[:], self.labels[:]= zip(*c)

    def next_batch(self, batch_size):  # 如果到达末尾，则把batch_size返回0，否则返回所读取的batch_size
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            return [], [], 0
        if (self.batch_id == 0):
            if self.is_shuffle == True:
                self._shuffle()
                if out_data==1:
                    with open(tempdatadir+"/init_train_data.txt",'w') as fout:
                        for item in train_data.data :
                            item=list(item)
                            for temp in item:
                                fout.write(str(temp))
                                fout.write(' ')
                            fout.write('\n')
                      
        end_id = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_id])
        batch_labels = (self.labels[self.batch_id:end_id])
        num=end_id-self.batch_id
        self.batch_id = end_id
        return batch_data, batch_labels,num 


def start_session_ckpt(model_dir):
    global_step = tf.Variable(0, name = 'global_step', trainable = False)   
    #freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点  
    #输出结点可以看我们模型的定义  
    #只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃  
    #所以,output_node_names必须根据不同的网络进行修改  

    # 加载模型参数
    print(' [*] Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(model_dir)        

    if ckpt and ckpt.model_checkpoint_path:
        #这里做了相对路径处理 比较方便移植
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1]\
                                                .split('-')[-1]
        print('Loading success, global_step is %s' % global_step)
    else:
        print(' [*] Failed to find a checkpoint')
        return -1
    # We import the meta graph and retrive a Saver   
    #  We clear the devices, to allow TensorFlow to 
    # control on the loading where it wants operations to be calculated  
    print('tensorflow version:%s'%str(tf.__version__))
    saver = tf.train.import_meta_graph(os.path.join(model_dir,ckpt_name) + '.meta',clear_devices=True)
    #creat session
    sess = tf.InteractiveSession()
    #init vars
    init = tf.group(tf.global_variables_initializer(), 
                    tf.local_variables_initializer())
    sess.run(init)
    #import saver vars
    saver.restore(sess, os.path.join(model_dir, ckpt_name)) 
    return sess   


# setting path

rootdir = '/data/mm0105.chen/wjhan/xiaomin'
tempdatadir='/data/mm0105.chen/wjhan/xiaomin/tempdata'
feadir = rootdir + '/feature'
logdir = rootdir + '/log'
modeldir = rootdir + '/model'
extra_train_set_path=[]#["/data/mm0105.chen/wjhan/dzy/LSTM/feature/IEMObyxiomin/denoise/IEMO.arff"]#["/data/mm0105.chen/wjhan/dzy/LSTM/feature/IEMObyxiomin/denoise/IEMO.arff"]
#"/data/mm0105.chen/wjhan/xiaomin/feature/intern_noise/intern_noise.arff","/data/mm0105.chen/wjhan/xiaomin/feature/iemo/washedS8/iemo.arff"
extra_val_set_path=""#"/data/mm0105.chen/wjhan/xiaomin/feature/intern_noise/intern_noise_all.arff"
#config




timestep_size = 1
corpus ='intern'
which_copy='byperson_denoise'
do_dropout=1
_keep_prob=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

gender_include = ['M','F']


person_exclude=['03','07','08','09','15','18']
scene_include=[]
scenario_include=[]
db_include=[]




# 每个隐含层的节点
hidden_size = [1024,512]
acc_train_epsilon= 0.98
epoch_num =128
_batch_size=1024
learning_rate = 0.003

# predefine

set_num = 29
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
print('******* Run LSTM %s ********' % (now.strftime('%Y-%m-%d %H:%M:%S')))
print('*********************************************')
print("SHUFFLE HAS REPEAT!")



#读入额外训练集
if len(extra_train_set_path)!=0 :
    for set_path in extra_train_set_path:
        extra_train_set=dr.ArffReader(extra_train_set_path)
        print(len(extra_train_set.data))


#读入额外验证集
if extra_val_set_path!="" :
    extra_val_set=dr.ArffReader(extra_val_set_path)

# read data
data_set = []
count=[]
for i in range(set_num):
    filepath = feadir+'/'+corpus+ '/' +which_copy+'/'+ str(i) + '.txt'
    with open(filepath, 'r') as fin:
        temp=fin.readlines()
        count.append(len(temp))
        data_set.append(temp)
# CV
'''
print(count)
print(len(count));
print(sum(count));
exit()
'''
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
 

    '''

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
            	print(temp[0])
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
    '''
    for j in range(set_num): # 29
        for item in data_set[j]:     # data_set 的大小为25， 有几个人data_set就有多大
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
            if j == i:
                val_set.append(fea)
                val_labels.append(onehot_label)
            else:
                train_set.append(fea)
                train_labels.append(onehot_label)
    if len(val_set) == 0 and i < 29:
        print('inner val_set size = 0, skip CV%d' %i)
        continue


    #add extra train data to train_set

    #额外训练集加入到train_set
    if len(extra_train_set_path)!=0 :
        for item in extra_train_set.data:
            temp=item.strip().split(',')
            name = temp[0].strip('\'').split('_')
            person=name[0][:2]
            if "Ses" in name[0]:
                fea = [float(index) for index in temp[1:-1]]
                emo_label=temp[-1]
                emo_label=temp[0].strip('\'').split('_')[-1]
                if emo_label not in emo_classes.keys():
                    continue
                onehot_label = np.zeros(class_num)
                onehot_label[emo_classes[emo_label]] = 1         
                train_set.append(fea)
                train_labels.append(onehot_label)
                continue                
            if "_C_" in temp[0]:
                fea = [float(index) for index in temp[1:-1]]
                emo_index=name[-1]
                if(emo_index=='4'):
                    emo_index=3
                onehot_label = np.zeros(class_num)
                onehot_label[int(emo_index)] = 1         
                train_set.append(fea)
                train_labels.append(onehot_label)
                continue
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
   


    #exit()

    val_set = np.array(val_set)
    val_labels = np.array(val_labels)
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    if out_data==1:
        with open(tempdatadir+"/ori_train_set.txt",'w') as fout:
            for item in train_set:           
                item=list(item)
                for temp in item:
                    fout.write(str(temp))
                    fout.write(' ')
                fout.write('\n')


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
    print('------额外训练集-----\n')
    print('scene_include:%s'%(scene_include))
    print('scenario_include:%s'%(scenario_include))
    print('db_include:%s'%(db_include))


    # pre-trained model load
    
    model_dir="/data/mm0105.chen/wjhan/xiaomin/model/2017-10-18_16_49_46_cv2"

    sess=start_session_ckpt(model_dir)

    graph=sess.graph
    #for op in graph.get_operations():
        #print(op.name)

    with open(tempdatadir+'/graph.txt','w') as fin:   
        for op in graph.get_operations():
            fin.write(str(op.name))
            fin.write('\n')
            fin.write(str(op.values()))

        #fin.write(str(sess.graph_def))
        #for op in sess.graph.get_operations():  
            #fin.write(op.name,op.values())  

    graph=sess.graph
    x = graph.get_tensor_by_name('input:0')  
    y = graph.get_tensor_by_name('predict:0')
    y_=graph.get_tensor_by_name('label:0')
    x_lengths=graph.get_tensor_by_name('x_lengths:0')
    batch_size=graph.get_tensor_by_name('batch_size:0')
    keep_prob=graph.get_tensor_by_name('keep_prob:0')

    optimizer=graph.get_operation_by_name('train_op')

    accuracy=graph.get_tensor_by_name('accuracy:0')
    cost=graph.get_tensor_by_name('cost:0')

   

    avg_cost = []
    acc_val = []
    acc_train = []


    if save_model == 1:
        saver = tf.train.Saver()
    acc_val_max=-1


    train_data = CDataSet(train_set, train_labels)

    #run epoch
    
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
    acc_val_cv.append(acc_val[np.argmax(acc_val)])
    acc_train_cv.append(acc_train[np.argmax(acc_val)])
    print('acc_train on cv %d = %f' % (i, acc_train_cv[-1]))
    print('acc_val on cv %d = %f' % (i, acc_val_cv[-1]))
    print('CV %d finish ' % (i))
    sess.close()
    tf.reset_default_graph()  
print('CV finished.\navg_acc_train=%f,avg_acc_val=%f' % (np.average(acc_train_cv), np.average(acc_val_cv)))
print('CV finished.\n avg_acc_val=%f'%np.average(acc_val_cv))
print('Train accuracy of each CV: %s'%str(acc_train_cv))
print('Val accuracy of each CV: %s'%str(acc_val_cv))
if output_log == 1:
    # 恢复标准输入输出
    sys.stdout = savedStdout
    fin.close()
print('Training finished.')
0.5,0.5,0.5,0.5,0.2,0.

if output_log == 1:
    # 恢复标准输入输出
    sys.stdout = savedStdout
    fin.close()
print('Training finished.')

