
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
        


def freeze_graph(model_dir):
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

    # We retrieve the protobuf graph definition  
    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
    with open('graph.txt','w') as fin:
        temp=sys.stdout
        sys.stdout=fin
        for op in graph.get_operations():  
            print(op.name,op.values())  
    sys.stdout=temp
    #fix batch norm cannot loaded by loadfreeze
    #Cannot convert a tensor of type float32 to an input of type float32_ref
    for node in input_graph_def.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

    # We use a built-in TF helper to export variables to constant  
    output_graph_def = graph_util.convert_variables_to_constants(  
        sess,   
        input_graph_def,   
        output_node_names#output_node_names.split(",") # We split on comma for convenience  
    )  
    
    freezef = tf.gfile.GFile(output_graph, "wb")
    freezef.write(output_graph_def.SerializeToString())  
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__=='__main__':
    output_node_names=['predict']
    model_dir = 'D:/xiaomin/pyproj/model_make/2017-07-11_19_31_33_cv0'
    freeze_graph(model_dir )