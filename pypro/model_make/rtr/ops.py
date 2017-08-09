# -*- coding: utf-8 -*-

import tensorflow as tf


def bias(name, shape, bias_start = 0.0, trainable = True):
    
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable, 
                          initializer = tf.constant_initializer(
                                                  bias_start, dtype = dtype))
    return var


def weight(name, shape, stddev = 0.02, trainable = True):
    
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable, 
                          initializer = tf.random_normal_initializer(
                                              stddev = stddev, dtype = dtype))
    return var


def OneHot(value, depth = 8):
    return tf.one_hot(value, depth, dtype = tf.float32)


def InnerProduct(value, output_shape, name = 'InnerProduct', with_w = False):
    
    shape = value.get_shape().as_list()
    
    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)
        
    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases

    
def lReLU(x, leak=0.2, name = 'lReLU'):
    
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x, name = name)
        
        
def ReLU(value, name = 'ReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)
    
    
def Deconv2d(value, output_shape, k_h = 5, k_w = 5, strides =[1, 2, 2, 1], 
             name = 'Deconv2d', with_w = False):
    
    with tf.variable_scope(name):
        weights = weight('weights', 
                         [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights, 
                                        output_shape, strides = strides)
        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv
            
            
def Conv2d(value, output_dim, k_h = 5, k_w = 5, 
            strides =[1, 2, 2, 1], name = 'Conv2d'):
    
    with tf.variable_scope(name):
        weights = weight('weights', 
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides = strides, padding = 'SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv


def MaxPooling(value, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], 
               padding = 'SAME', name = 'MaxPooling'):
                   
    with tf.variable_scope(name):
        return tf.nn.max_pool(value, ksize = ksize, 
                              strides = strides, padding = padding)


def Concat(value, cond, name = 'concat'):
    
    """
    Concatenate conditioning vector on feature map axis.
    """
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    
    with tf.variable_scope(name):        
        return tf.concat([value, 
              cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], axis = 3)


def BatchNorm(value, is_train = True, name = 'BatchNorm', 
               epsilon = 1e-5, momentum = 0.9):
    
    return tf.contrib.layers.batch_norm(value, decay = momentum, 
                                        updates_collections = None, 
                                        epsilon = epsilon, scale = True, 
                                        is_training = is_train, scope = name)