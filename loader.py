
import tensorflow as tf
import numpy as np  

import os
import sys, getopt

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
            producer_op_list=None  
        )  
    return graph  


def read_fea(feapath):
    with open(feapath,'r') as fin:
        data=fin.readlines()
        return data[1].strip().split(';')[2:]


fea_path='D:/xiaomin/pyproj/model_make/output_segment_0003_done.csv'

frozen_graph_filename='D:/xiaomin/pyproj/model_make/frozen_model.pb'

data=read_fea(fea_path)

graph=load_graph(frozen_graph_filename)

for op in graph.get_operations():  
    print(op.name,op.values())    

x = graph.get_tensor_by_name('input:0')  
y = graph.get_tensor_by_name('predict:0')

with tf.Session(graph=graph) as sess:
    probability = sess.run(y, feed_dict = {x:[data]})
    print(probability)