# -*- coding: utf-8 -*-
"""
@created: Taorui ren 
主要是为了生成一输入对一输出的模型
运行 example
python prefreeze.py -f testimage/test_001.png
"""

import os
import numpy as np
import pprint
import tensorflow as tf
from PIL import Image
import sys, getopt

from ops import Conv2d, BatchNorm, lReLU, InnerProduct

CURRENT_DIR = os.getcwd()
#参数处理
opts, args = getopt.getopt(sys.argv[1:], "f:")
test_face_path=""
for op, value in opts:
    if op == "-f":
        test_face_path = value
    else:
        print -1;   
        sys.exit()



def discriminator(image, num_of_data,reuse = False, name = 'discriminator'):
    
    with tf.variable_scope(name):    
    
        if reuse:
            tf.get_variable_scope().reuse_variables()        
            
        h1 = lReLU(BatchNorm(Conv2d(image, 64, name='d_Conv2d1'), 
                             name = 'd_bn1'))        
        h2 = lReLU(BatchNorm(Conv2d(h1, 64*2, name='d_Conv2d2'), 
                             name = 'd_bn2'))            
        h3 = lReLU(BatchNorm(Conv2d(h2, 64*4, name='d_Conv2d3'), 
                             name = 'd_bn3'))
        h4 = lReLU(BatchNorm(Conv2d(h3, 64*8, name='d_Conv2d4'), 
                             name = 'd_bn4'))
        shared = InnerProduct(tf.reshape(h4, [num_of_data, -1]), 
                              1024, name = 'd_fc_shared')
#        disc = InnerProduct(shared, 1, 'd_fc_disc')
        
        recog_shared = InnerProduct(shared, 128, name = 'd_fc_recog_shared')
        cat = InnerProduct(recog_shared, 8)
        #@taorui Just for output predict result
        emo_predict = tf.nn.softmax(cat, dim=1, name = 'emo_predict')
        return cat
        
def getemo(test_face_path):
    #文件不存在异常处理
    if not os.path.exists(test_face_path):
        print('inputs face not exists')
        return -1
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    model_dir = CURRENT_DIR + '/model/' + '01/'
    predict_model_path = CURRENT_DIR + '/model/predict/'
    
    if not os.path.exists(model_dir):
        print('train model not exists')
        return -1

    if not os.path.exists(predict_model_path):
        os.makedirs(predict_model_path)

    #placeholder set [1.*.*.1] for one test image
    test_images = tf.placeholder(tf.float32, [1, 128, 128, 1],name='input_image')

    #test discriminator
    test_cat = discriminator(test_images,1)

    test_result = tf.argmax(test_cat, axis = 1)
    test_probability = tf.nn.softmax(test_cat,dim = 1,name="test_probability")

    #init vars
    init = tf.group(tf.global_variables_initializer(), 
                    tf.local_variables_initializer())

    if os.environ.get('CUDA_VISIBLE_DEVICES') == None:
        os.environ['CUDA_VISIBLE_DEVICES'] ='14' #str(view_gpu.GPU_ID)
        print "set env CUDA_VISIBLE_DEVICES = 14 "
    else:
        print("CUDA_VISIBLE_DEVICES is already set as %s\n" % (os.environ.get('CUDA_VISIBLE_DEVICES')))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    sess.run(init)
    coord = tf.train.Coordinator()
    print("begin to train ====================>")
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    saver = tf.train.Saver(max_to_keep = None) 
    print(' [*] Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(model_dir)        

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        global_step = ckpt.model_checkpoint_path.split('/')[-1]\
                                                .split('-')[-1]
        print('Loading success, global_step is %s' % global_step)
    else:
        print(' [*] Failed to find a checkpoint')
        return -1

    #range just for test model run time
    #you can set range(0,100) to test 
    for idx in range(0,1):
        #loadtest data
        ori_face = [ Image.open(test_face_path).convert('L').resize((128,128)) ]
        test_face = [ np.asarray(img, dtype='float32').reshape(128,128,1)/256 for img in ori_face ]
        result = sess.run(test_result, feed_dict = {test_images: test_face})[0]
        probability = sess.run(test_probability,feed_dict = {test_images: test_face})
        print("[<--%d-->]"%(result))
        print probability


    print("Save predict model!---------->");
    saver.save(sess, os.path.join(predict_model_path,'predict_model.ckpt'))

if __name__ == '__main__':
    if not os.path.exists(test_face_path):
        print('inputs face not exists')
        sys.exit()
    getemo(test_face_path)

        
