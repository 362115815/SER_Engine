import os
import numpy as np
import pprint
import tensorflow as tf
import sys


from tensorflow.python.framework import graph_util

CURRENT_DIR = os.getcwd()

def freeze_graph(model_dir):
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
    saver = tf.train.import_meta_graph(os.path.join(model_dir,ckpt_name) + '.meta', clear_devices=True)

    #creat session
    sess = tf.InteractiveSession()
    #init vars
    init = tf.group(tf.global_variables_initializer(), 
                    tf.local_variables_initializer())
    sess.run(init)
    #import saver vars
    saver.restore(sess, os.path.join(model_dir, ckpt_name)) 
    print(ckpt_name)

if __name__=='__main__':
    output_node_names=['predict']
    model_dir = 'D:/xiaomin/pyproj/model_make/2017-08-08_08_42_49_cv0'
    freeze_graph(model_dir )