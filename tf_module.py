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
            producer_op_list=None  
        )  
    return graph  


def start_session(graph):
    return tf.Session(graph=graph)


def close_session(sess):
    sess.close()


def run(sess,fea_path,outpath):
    x=sess.graph.get_tensor_by_name('input:0')
    y=sess.graph.get_tensor_by_name('predict:0')
    trial_data=read_fea(fea_path)
    predict=sess.run(y,feed_dict={x:[trial_data]})
    index=np.argmax(predict)
    with open(outpath,'w') as fout:
        fout.write(emo_classes[index])
    return predict
def run_y(fea_path):
    return read_fea(fea_path)

def get_predict():
    return predict

