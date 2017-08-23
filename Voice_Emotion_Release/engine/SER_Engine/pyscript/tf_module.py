#!/mnt/sda1/hushuo/anaconda3/bin/python
# encoding: utf-8
import tensorflow as tf
import numpy as np
import os
import sys, getopt
predict=[]
emo_classes={0:'anger',1:'elation',2:'neutral',3:'panic',4:'sadness'}

def read_fea(feapath):
    with open(feapath,'r') as fin:
        data=fin.readlines()
        return data[1].strip().split(';')[2:]


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
        )
    return graph



def start_session(graph):
    return tf.Session(graph=graph)


def close_session(sess):
    sess.close()


def run(sess,fea_path,outpath):
    print("run run run")
    x=sess.graph.get_tensor_by_name('input:0')
    y=sess.graph.get_tensor_by_name('predict:0')
    trial_data=read_fea(fea_path)
    predict=sess.run(y,feed_dict={x:[trial_data]})
    index=np.argmax(predict)
    with open(outpath,'w') as fout:
        fout.write(emo_classes[index]+'\n')
        for item in predict:
            for i in range(len(item)-1):
                fout.write(str(item[i])+" ")
            fout.write(str(item[-1])+'\n')


def run_lstm(sess,fea_path,outpath):
    #print("run run run lstm")
    graph = tf.get_default_graph()  
    x = graph.get_tensor_by_name('input:0')  
    y = graph.get_tensor_by_name('predict:0')
    x_lengths=graph.get_tensor_by_name('x_lengths:0')
    batch_size=graph.get_tensor_by_name('batch_size:0')
    keep_prob=graph.get_tensor_by_name('keep_prob:0')
    trial_data=read_fea(fea_path)
    predict=sess.run(y,feed_dict={x:[trial_data],x_lengths:[1],batch_size:1,keep_prob:1.0})
    predict=predict.tolist()

    temp=predict[0][3]
    predict[0][3]=0
    predict[0].append(temp)
    #print(predict)
    #print(np.shape(predict)) 
    index=np.argmax(predict)
    with open(outpath,'w') as fout:
        fout.write(emo_classes[index]+'\n')
        for item in predict:
            for i in range(len(item)-1):
                fout.write(str(item[i])+" ")
            fout.write(str(item[-1])+'\n')    
    return predict

def run_y(fea_path):
    return read_fea(fea_path)

def get_predict():
    return predict




def start_session_ckpt(model_dir):
    #model_dir='D:/xiaomin/pyproj/model_make/2017-08-10_14_30_51_cv3'
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