
import os
import numpy as np
import pprint
import tensorflow as tf
from PIL import Image
import sys
#from ops import Conv2d, BatchNorm, lReLU, InnerProduct
#加载将参数和结构图同时保存的包
from tensorflow.python.framework import graph_util

CURRENT_DIR = os.getcwd()
        

def load_graph(model_dir):
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    
    #freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点  
    #输出结点可以看我们模型的定义  
    #只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃  
    #所以,output_node_names必须根据不同的网络进行修改  

    output_graph = model_dir + "/frozen_model.pb"  
    
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
    saver = tf.train.import_meta_graph(os.path.join(model_dir,ckpt_name) + '.meta', clear_devices=True)

    if os.environ.get('CUDA_VISIBLE_DEVICES') == None:
        os.environ['CUDA_VISIBLE_DEVICES'] ='14' #str(view_gpu.GPU_ID)
        #print "set env CUDA_VISIBLE_DEVICES = 14 "
    else:
        print("CUDA_VISIBLE_DEVICES is already set as %s\n" % (os.environ.get('CUDA_VISIBLE_DEVICES')))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #creat session
    sess = tf.InteractiveSession(config = config)
    #init vars
    init = tf.group(tf.global_variables_initializer(), 
                    tf.local_variables_initializer())
    sess.run(init)

    #import saver vars
    saver.restore(sess, os.path.join(model_dir, ckpt_name)) 

    return sess

def read_fea(feapath):
    with open(feapath,'r') as fin:
        data=fin.readlines()
        return data[1].strip().split(';')[2:]

if __name__=='__main__':
    output_node_names=['predict']
    model_dir = 'D:/xiaomin/pyproj/model_make/2017-08-08_09_37_02_cv0'
    sess=load_graph(model_dir )

    fea_path='D:/xiaomin/pyproj/model_make/output_segment_0003_done.csv'
    data=read_fea(fea_path)
    graph = tf.get_default_graph()  
    x = graph.get_tensor_by_name('input:0')  
    y = graph.get_tensor_by_name('predict:0')
    x_lengths=graph.get_tensor_by_name('x_lengths:0')
    batch_size=graph.get_tensor_by_name('batch_size:0')
    keep_prob=graph.get_tensor_by_name('keep_prob:0')

    probability = sess.run(y, feed_dict = {x:[data],x_lengths:[1],batch_size:1,keep_prob:1.0})
    print(probability)

