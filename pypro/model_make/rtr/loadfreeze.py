# -*- coding: utf-8 -*-
"""
@created: Taorui ren 
主要是为了测试生成一输入对一输出的模型
运行 example
python loadfreeze.py -f testimage/test_001.png
"""
import tensorflow as tf
import numpy as np  
from PIL import Image
import os
import sys, getopt

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
            name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  
  
if __name__ == '__main__':  
    
    frozen_model_filename = os.path.join(CURRENT_DIR,'model/predict/frozen_model.pb')
    #加载已经将参数固化后的图  
    graph = load_graph(frozen_model_filename)  

    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    for op in graph.get_operations():  
        print(op.name,op.values())    
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字  
    x = graph.get_tensor_by_name('prefix/input_image:0')  
    y = graph.get_tensor_by_name('prefix/discriminator/emo_predict:0')  
    
    if not os.path.exists(test_face_path):
        print('inputs face not exists')
        sys.exit()
    
    #init config 
    if os.environ.get('CUDA_VISIBLE_DEVICES') == None:
        os.environ['CUDA_VISIBLE_DEVICES'] ='14' #str(view_gpu.GPU_ID)
        print "set env CUDA_VISIBLE_DEVICES = 14 "
    else:
        print("CUDA_VISIBLE_DEVICES is already set as %s\n" % (os.environ.get('CUDA_VISIBLE_DEVICES')))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #loadtest data & predict
    ori_face = [ Image.open(test_face_path).convert('L').resize((128,128)) ]
    test_face = [ np.asarray(img, dtype='float32').reshape(128,128,1)/256 for img in ori_face ]
    with tf.Session(graph=graph) as sess:
        probability = sess.run(y, feed_dict = {x: test_face})[0]
        print probability
